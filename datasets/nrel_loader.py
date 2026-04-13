# Data loader for NREL dataset

import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFile
import pvlib
from datetime import datetime, timedelta
import hashlib
import json
from tqdm import tqdm

from .chrono import filter_frame_to_windows, renormalize_on_window, resolve_split_windows, split_samples_by_windows

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _select_consecutive_year_block(available_years, window=3):
    years = sorted(int(year) for year in available_years)
    for start in range(len(years) - window + 1):
        candidate = years[start : start + window]
        if candidate == list(range(candidate[0], candidate[0] + window)):
            return candidate
    raise ValueError(f"Unable to find {window} consecutive benchmark years in {years}.")


def _chronological_benchmark_split(samples, val_split):
    # For fair Swin/ConvNeXt baselines, we truncate NREL to the earliest 3-year
    # chronological window: first 2 years for train/validation, next year for test.
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
        self.sequence_length, self.sampling_rate = config['data']['sequence_length'], config['data']['sampling_rate_sec']
        self.image_size = int(config['data'].get('image_size', 224))
        self.horizons = config['model']['horizons'] 
        self.csv_path = config['data'].get('csv_path')
        self.image_root = config['data'].get('image_root')
        self.use_aux = config['model'].get('use_aux_decoder', False)
        self.use_benchmark_split = bool(config['data'].get('enforce_benchmark_split', True))
        self.split_windows = resolve_split_windows(config)

        self.df = self._load_all_data()
        if self.split_windows is not None:
            self.df = filter_frame_to_windows(self.df, self.split_windows)
        elif config['data'].get('years') and not self.use_benchmark_split:
            self.df = self.df[self.df.index.year.isin(config['data']['years'])]

        # NREL Coordinates
        self.lat, self.lon, self.alt = 39.742, -105.178, 1828.8 
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
        try: df['Datetime'] = pd.to_datetime(df[col], format='%m/%d/%Y %H:%M')
        except: df['Datetime'] = pd.to_datetime(df[col], infer_datetime_format=True)
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

    def _build_full_image_index(self):
        year_scope = 'benchmark_all' if self.use_benchmark_split else self.config['data'].get('years', 'all')
        cache_conf = {'root': self.image_root, 'tol': self.config['data'].get('image_tolerance_sec', 120), 'years': year_scope}
        h = hashlib.md5(json.dumps(cache_conf, sort_keys=True).encode()).hexdigest()[:8]
        cache_path = os.path.join(self.image_root, f"_image_index_cache_{h}.pkl")
        
        if os.path.exists(cache_path) and not self.config['data'].get('rebuild_image_cache', False):
            with open(cache_path, 'rb') as f: return pickle.load(f)
        if not self.config['data'].get('rebuild_image_cache', False):
            existing_caches = sorted(
                path for path in os.listdir(self.image_root)
                if path.startswith("_image_index_cache_") and path.endswith(".pkl")
            )
            if existing_caches:
                fallback_cache = os.path.join(self.image_root, existing_caches[0])
                print(f"Reusing existing NREL image index cache: {fallback_cache}")
                with open(fallback_cache, 'rb') as f:
                    return pickle.load(f)

        print("Building NREL image index (10-min cadence) target=_11.jpg...")
        recs = []
        cands = [os.path.join(r, n) for r, d, fs in os.walk(self.image_root) for n in fs if n.endswith('_11.jpg')]
        for p in tqdm(cands, desc="Validating"):
            try:
                ts = datetime.strptime(os.path.basename(p)[:14], "%Y%m%d%H%M%S")
                with Image.open(p) as img: img.verify()
                recs.append({'timestamp': ts, 'filepath': p})
            except: continue

        idx_df = pd.DataFrame(recs).sort_values('timestamp').reset_index(drop=True)
        with open(cache_path, 'wb') as f: pickle.dump(idx_df, f)
        return idx_df

    def _match_images(self):
        img_idx = self._build_full_image_index()
        w_df = pd.DataFrame({'timestamp': self.df.index}).sort_values('timestamp').reset_index(drop=True)
        
        matched = pd.merge_asof(w_df, img_idx, on='timestamp', direction='nearest', tolerance=pd.Timedelta(seconds=self.config['data'].get('image_tolerance_sec', 300))).dropna()
        # Create a fast lookup dictionary from timestamp to filepath
        img_dict = dict(zip(matched['timestamp'], matched['filepath']))
        
        df_set = set(self.df.index)
        samples = []
        
        for dt, img_path in img_dict.items():
            dt = dt.to_pydatetime()
            if (dt - timedelta(seconds=self.sampling_rate * (self.sequence_length - 1))) not in df_set: continue
            if not all((dt + timedelta(minutes=h)) in df_set for h in self.horizons): continue
            
            if self.use_aux:
                next_dt = pd.Timestamp(dt + timedelta(seconds=self.sampling_rate))
                if next_dt in img_dict:
                    samples.append((img_path, img_dict[next_dt], dt))
            else:
                samples.append((img_path, None, dt))
                
        print(f"Final valid NREL samples: {len(samples)}")
        return samples

    def __len__(self): return len(self.samples)

    def _process_image(self, path, size=None, mask_radius=None, is_target=False):
        size = size or (self.image_size, self.image_size)
        mask_radius = max(1, int(round(min(size) * 250 / 512))) if mask_radius is None else mask_radius
        img = Image.open(path).convert('RGB').resize(size)
        mask = Image.new('L', size, 0)
        c = (size[0]//2, size[1]//2)
        ImageDraw.Draw(mask).ellipse((c[0]-mask_radius, c[1]-mask_radius, c[0]+mask_radius, c[1]+mask_radius), fill=255)
        img_np = np.array(img)
        img_np[np.array(mask) == 0] = 0
        img = Image.fromarray(img_np)
        
        if is_target:
            img = img.resize((128, 128))
            return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)
        return self.transform(img) if self.transform else img

    def __getitem__(self, idx):
        # Retry logic for NREL
        for att in range(5):
            try:
                img_p, next_p, dt = self.samples[(idx + att) % len(self.samples)]
                image = self._process_image(img_p)
                next_image = self._process_image(next_p, is_target=True) if self.use_aux else None
                break
            except: continue

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
    val_split = float(config['training'].get('val_split', 0.1))

    if ds.split_windows is not None:
        tr_idx, v_idx, t_idx, split_summary = split_samples_by_windows(ds.samples, ds.split_windows, val_split=val_split)
        renormalize_on_window(ds, ds.split_windows.trainval_start, ds.split_windows.trainval_end)
        print(f"[split][nrel] {split_summary}")
    else:
        tr_idx, v_idx, t_idx, selected_years = _chronological_benchmark_split(
            ds.samples,
            val_split=val_split,
        )
        _renormalize_on_trainval_years(ds, trainval_years=set(selected_years[:2]))
        print(f"[split][nrel] train/val years={selected_years[:2]} test year={selected_years[2]}")

    train_stride = max(1, int(config['data'].get('train_stride', 1)))
    val_stride = max(1, int(config['data'].get('val_stride', 1)))
    test_stride = max(1, int(config['data'].get('test_stride', 1)))
    tr_idx = tr_idx[::train_stride]
    v_idx = v_idx[::val_stride]
    t_idx = t_idx[::test_stride]

    bs, nw = config['data']['batch_size'], config['data']['num_workers']
    loader_kwargs = {
        'num_workers': nw,
        'pin_memory': torch.cuda.is_available(),
    }
    if nw > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = int(config['data'].get('prefetch_factor', 4))

    return (
        DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=bs, shuffle=True, **loader_kwargs),
        DataLoader(torch.utils.data.Subset(ds, v_idx), batch_size=bs, shuffle=False, **loader_kwargs),
        DataLoader(torch.utils.data.Subset(ds, t_idx), batch_size=bs, shuffle=False, **loader_kwargs),
    )
