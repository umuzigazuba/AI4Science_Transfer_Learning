import torch
import lightning

import polars as pl


def pad_lightcurves(lcs_parquet, columns_to_pad = ["TIME", "FLUX", "FLUXERR", "BAND"]):

    maxlen = max(lcs_parquet["TIME"].list.len())

    for col in columns_to_pad:
        
        lcs_parquet = lcs_parquet.with_columns((pl.col(col)).alias(col + "_PAD"))
        
        lcs_parquet = lcs_parquet.with_columns(pl.col(col + "_PAD").list.concat(pl.lit(0).repeat_by(maxlen - pl.col(col).list.len())))
         
    lcs_parquet = lcs_parquet.with_columns(pl.lit(False).repeat_by(pl.col(col).list.len()).list.concat(pl.lit(True).repeat_by(maxlen - pl.col(col).list.len())).alias("MASK_PAD"))

    return lcs_parquet

def reformat_bands(lcs_parquet):
    
    band_dic = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "y": 5, "0": -1}

    lcs_parquet = lcs_parquet.with_columns(pl.col("BAND_PAD").map_elements(lambda bands: pl.Series([band_dic.get(b, -1) for b in bands], 
                                                                                                   dtype = pl.Int32), 
                                                                                                   return_dtype = pl.List(pl.Int32),
                                                                          ).alias("BAND_NUMBER"))
    return lcs_parquet

class Augmented_Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, file_path): 

        self.lcs_parquet = pl.read_parquet(file_path)
        self.lcs_parquet = pad_lightcurves(self.lcs_parquet)
        self.lcs_parquet = reformat_bands(self.lcs_parquet)

    def __len__(self):

        return len(self.lcs_parquet)

    def __getitem__(self, idx):

        time = torch.tensor(self.lcs_parquet["TIME_PAD"][idx].to_numpy().flatten())
        mask_pad = torch.tensor(self.lcs_parquet["MASK_PAD"][idx].to_numpy().flatten())
        time[~mask_pad] = time[~mask_pad] - torch.min(time[~mask_pad])

        flux = torch.tensor(self.lcs_parquet["FLUX_PAD"][idx].to_numpy().flatten())
        flux_err = torch.tensor(self.lcs_parquet["FLUXERR_PAD"][idx].to_numpy().flatten())
        band = torch.tensor(self.lcs_parquet["BAND_NUMBER"][idx].to_numpy())

        redshift = torch.tensor(self.lcs_parquet["Z"][idx])
        ebv = torch.tensor(self.lcs_parquet["EBV"][idx])

        data = torch.stack([time, band, flux, flux_err]).to(torch.float32)
        meta = torch.stack([redshift, ebv])
        target = torch.tensor(self.lcs_parquet["target"][idx])

        return data, meta, target

class LightCurveModule(lightning.LightningDataModule):

    def __init__(self, file_path: str, validation_fraction: float, train_batch_size: int, 
                 validation_batch_size: int, num_workers: int = 9):

        super().__init__()

        self.file_path = file_path
        self.validation_fraction = validation_fraction

        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):

        if not hasattr(self, 'train_dataset'):
            self.dataset = Augmented_Dataset(self.file_path)
            
            len_dataset = len(self.dataset)
            len_validation_dataset = int(len_dataset * self.validation_fraction)

            self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
                self.dataset, 
                [len_dataset - len_validation_dataset, len_validation_dataset])
        
    def train_dataloader(self) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.train_batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            persistent_workers = True,
            pin_memory = True,  
            generator = torch.Generator(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size = self.validation_batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            persistent_workers = True,
            pin_memory = True,  
            generator = torch.Generator(),
        )


