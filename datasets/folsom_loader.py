# Data loader for Folsom dataset

import os
import bisect
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFile
import pvlib
from datetime import datetime, timedelta

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _select_consecutive_year_block(available_years, window=3):
    years = sorted(int(year) for year in available_years)
    for start in range(len(years) - window + 1):
        candidate = years[start : start + window]
        if candidate == list(range(candidate[0], candidate[0] + window)):
            return candidate
    raise ValueError(f"Unable to find {window} consecutive benchmark years in {years}.")


def _chronological_benchmark_split(samples, val_split):
    # Fair Swin/ConvNeXt-style benchmarking keeps a fixed chronological boundary:
    # earliest 2 years for train/validation, and the subsequent year held out for test.
    available_years = sorted({dt.year for _, _, dt in samples})
    selected_years = _select_consecutive_year_block(available_years, window=3)
    trainval_years = set(selected_years[:2])
    test_year = selected_years[2]

    trainval_idx = [i for i, (_, _, dt) in enumerate(samples) if dt.year in trainval_years]
    test_idx = [i for i, (_, _, dt) in enumerate(samples) if dt.year == test_year]
    if len(trainval_idx) < 2 or not test_idx:
        raise RuntimeError("Chronological benchmark split produced an empty partition.")

    val_len = max(1, int(len(trainval_idx) * val_split))
    val_len = min(val_len, len(trainval_idx) - 1)
    train_idx = trainval_idx[:-val_len]
    val_idx = trainval_idx[-val_len:]
    return train_idx, val_idx, test_idx, selected_years


def _renormalize_on_trainval_years(dataset, trainval_years):
    cols_to_norm = ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth']
    restored = dataset.df[dataset.feature_cols].copy()
    restored[cols_to_norm] = restored[cols_to_norm] * (dataset.std[cols_to_norm] + 1e-6) + dataset.mean[cols_to_norm]

    stats_frame = restored[restored.index.year.isin(trainval_years)]
    dataset.mean = stats_frame[dataset.feature_cols].mean()
    dataset.std = stats_frame[dataset.feature_cols].std()
    dataset.df[cols_to_norm] = (restored[cols_to_norm] - dataset.mean[cols_to_norm]) / (dataset.std[cols_to_norm] + 1e-6)

class SolarDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        self.root_dir = config['data']['server_root'] if config['env'] == 'server' else config['data']['local_root']
        self.sequence_length = config['data']['sequence_length']
        self.sampling_rate = config['data']['sampling_rate_sec']
        self.horizons = config['model']['horizons'] 
        self.csv_path = config['data'].get('csv_path')
        self.image_root = config['data'].get('image_root')
        self.image_tolerance = timedelta(seconds=config['data'].get('image_tolerance_sec', 120))
        self.use_aux = config['model'].get('use_aux_decoder', False)
        self.use_benchmark_split = bool(config['data'].get('enforce_benchmark_split', True))

        self.df = self._load_all_data()
        if config['data'].get('years') and not self.use_benchmark_split:
            self.df = self.df[self.df.index.year.isin(config['data']['years'])]

        self.lat, self.lon, self.alt = 38.642, -121.148, 60.0 
        self.df = self._add_solar_physics(self.df, self.lat, self.lon, self.alt)
        self.df = self.df[self.df['SZA'] <= 85]

        self.feature_cols = ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth', 'sin_hour', 'cos_hour']
        self.mean, self.std = self.df[self.feature_cols].mean(), self.df[self.feature_cols].std()
        
        cols_to_norm = ['k_index', 'temperature', 'pressure', 'SZA', 'Azimuth']
        self.df[cols_to_norm] = (self.df[cols_to_norm] - self.mean[cols_to_norm]) / (self.std[cols_to_norm] + 1e-6)

        self.samples = self._match_images()

    def _load_all_data(self):
        df = pd.read_csv(self.csv_path)
        col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df['Datetime'] = pd.to_datetime(df[col]) if col == 'Datetime' else pd.to_datetime(df[col], format='%Y%m%d%H%M%S')
        master_df = df.sort_values('Datetime').set_index('Datetime')
        for c in ['GHI', 'DNI', 'DHI', 'temperature', 'pressure']: master_df[c] = pd.to_numeric(master_df[c], errors='coerce')
        return master_df.dropna()

    def _add_solar_physics(self, df, lat, lon, alt):
        site = pvlib.location.Location(lat, lon, altitude=alt)
        sp = site.get_solarposition(df.index)
        df['SZA'], df['Azimuth'] = sp['zenith'], sp['azimuth']
        df['GHI_cs'] = site.get_clearsky(df.index, model='ineichen')['ghi']
        df['k_index'] = (df['GHI'] / (df['GHI_cs'] + 1e-6)).clip(0.0, 1.2)
        df['hour'] = df.index.hour + df.index.minute / 60.0
        df['sin_hour'], df['cos_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0), np.cos(2 * np.pi * df['hour'] / 24.0)
        return df

    def _build_day_image_index(self, day_dir):
        if not os.path.isdir(day_dir): return None
        ts_list, files = [], []
        for name in os.listdir(day_dir):
            if not name.lower().endswith('.jpg'): continue
            try: ts_list.append(datetime.strptime(os.path.splitext(name)[0], "%Y%m%d_%H%M%S"))
            except: continue
            files.append(name)
        if not ts_list: return None
        order = np.argsort(ts_list)
        return {"dir": day_dir, "timestamps": [ts_list[i] for i in order], "files": [files[i] for i in order]}

    def _closest_image(self, dt, cache):
        d_key = dt.date()
        if d_key not in cache:
            cache[d_key] = self._build_day_image_index(os.path.join(self.image_root, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")))
        idx = cache[d_key]
        if not idx: return None
        
        pos = bisect.bisect_left(idx["timestamps"], dt)
        cands = []
        if pos < len(idx["timestamps"]): cands.append((idx["timestamps"][pos], idx["files"][pos]))
        if pos > 0: cands.append((idx["timestamps"][pos-1], idx["files"][pos-1]))
        if not cands: return None
        
        best_ts, best_f = min(cands, key=lambda x: abs(x[0] - dt))
        if abs(best_ts - dt) <= self.image_tolerance: return os.path.join(idx["dir"], best_f)
        return None

    def _match_images(self):
        samples, cache = [], {}
        for dt in self.df.index:
            if (dt - timedelta(seconds=self.sampling_rate * (self.sequence_length - 1))) not in self.df.index: continue
            if not all((dt + timedelta(minutes=h)) in self.df.index for h in self.horizons): continue
            
            img_path = self._closest_image(dt, cache)
            if not img_path: continue
            
            if self.use_aux:
                next_dt = dt + timedelta(seconds=self.sampling_rate)
                next_path = self._closest_image(next_dt, cache)
                if not next_path: continue
                samples.append((img_path, next_path, dt))
            else:
                samples.append((img_path, None, dt))
        return samples

    def __len__(self): return len(self.samples)

    def _process_image(self, path, size=(512, 512), mask_radius=250, is_target=False):
        img = Image.open(path).convert('RGB').resize(size)
        mask = Image.new('L', size, 0)
        c = (size[0]//2, size[1]//2)
        ImageDraw.Draw(mask).ellipse((c[0]-mask_radius, c[1]-mask_radius, c[0]+mask_radius, c[1]+mask_radius), fill=255)
        img_np = np.array(img)
        img_np[np.array(mask) == 0] = 0
        img = Image.fromarray(img_np)
        
        if is_target: # Low res for UNet Target
            img = img.resize((128, 128))
            return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)
        return self.transform(img) if self.transform else img

    def __getitem__(self, idx):
        img_p, next_p, dt = self.samples[idx]
        
        image = self._process_image(img_p)
        if self.use_aux: next_image = self._process_image(next_p, is_target=True)

        seq_ts = [dt - timedelta(seconds=i*self.sampling_rate) for i in range(self.sequence_length)][::-1]
        w_slice = self.df[self.feature_cols].reindex(seq_ts, method='nearest', tolerance=pd.Timedelta('10min')).ffill().bfill().fillna(0.0)
        
        w_vals = w_slice.values
        if w_vals.dtype == 'object': w_vals = w_slice.apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        weather_seq = torch.tensor(w_vals, dtype=torch.float32)

        t_vals, cs_vals = [], []
        for h in self.horizons:
            r = self.df.loc[dt + timedelta(minutes=h)]
            t_vals.append(r['k_index'].iloc[0] if isinstance(r['k_index'], pd.Series) else r['k_index'])
            cs_vals.append(r['GHI_cs'].iloc[0] if isinstance(r['GHI_cs'], pd.Series) else r['GHI_cs'])
            
        targets, ghi_cs = torch.tensor(t_vals, dtype=torch.float32), torch.tensor(cs_vals, dtype=torch.float32)

        if self.use_aux: return image, weather_seq, targets, ghi_cs, next_image
        return image, weather_seq, targets, ghi_cs

def get_data_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = SolarDataset(config, transform=transform)

    tr_idx, v_idx, t_idx, selected_years = _chronological_benchmark_split(
        ds.samples,
        val_split=float(config['training'].get('val_split', 0.1)),
    )
    _renormalize_on_trainval_years(ds, trainval_years=set(selected_years[:2]))
    print(f"[split][folsom] train/val years={selected_years[:2]} test year={selected_years[2]}")

    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    v_ds = torch.utils.data.Subset(ds, v_idx)
    t_ds = torch.utils.data.Subset(ds, t_idx)

    bs, nw = config['data']['batch_size'], config['data']['num_workers']
    return DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw), DataLoader(v_ds, batch_size=bs, shuffle=False, num_workers=nw), DataLoader(t_ds, batch_size=bs, shuffle=False, num_workers=nw)
