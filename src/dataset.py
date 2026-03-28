from torch.utils.data import Dataset, DataLoader

from src.chains import sample_random_specs, stitch_segments, center_coords


class SyntheticChainDataset(Dataset):
    def __init__(
        self,
        num_samples,
        total_len=64,
        bond_len=1.0,
        min_segments=3,
        max_segments=5,
        min_len=12,
        max_len=24,
        random_orient=True,
        center=True,
        device="cpu",
    ):
        self.num_samples = num_samples
        self.total_len = total_len
        self.bond_len = bond_len
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.min_len = min_len
        self.max_len = max_len
        self.random_orient = random_orient
        self.center = center
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        specs = sample_random_specs(
            total_len=self.total_len,
            min_segments=self.min_segments,
            max_segments=self.max_segments,
            min_len=self.min_len,
            max_len=self.max_len,
        )

        coords, labels, seg_ids = stitch_segments(
            specs,
            bond_len=self.bond_len,
            random_orient=self.random_orient,
            device=self.device,
        )

        if self.center:
            coords = center_coords(coords)

        return {
            "coords": coords,
            "labels": labels,
            "seg_ids": seg_ids,
        }


def create_dataloader(num_samples, batch_size, total_len=64, bond_len=1.0, **kwargs):
    ds = SyntheticChainDataset(
        num_samples=num_samples,
        total_len=total_len,
        bond_len=bond_len,
        **kwargs,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
