import numpy as np
import h5py
import mne
from braindecode.datasets.bbci import BBCIDataset


class PatchedBBCIDataset(BBCIDataset):
    def _add_markers(self, cnt):
        with h5py.File(self.filename, "r") as h5file:
            event_times_in_ms = h5file["mrk"]["time"][:].squeeze()
            event_classes = h5file["mrk"]["event"]["desc"][:].squeeze().astype(np.int64)
            class_name_set = h5file["nfo"]["className"][:].squeeze()
            all_class_names = [
                "".join(chr(c.item()) for c in h5file[obj_ref])
                for obj_ref in class_name_set
            ]

        event_times_in_samples = np.uint32(
            np.round(event_times_in_ms * cnt.info["sfreq"] / 1000.0)
        )

        stim_chan = np.zeros(cnt.n_times, dtype=np.int64)
        for i_sample, id_class in zip(event_times_in_samples, event_classes):
            if i_sample < cnt.n_times:
                stim_chan[i_sample] += id_class

        stim_info = mne.create_info(
            ch_names=["STI 014"],
            sfreq=cnt.info["sfreq"],
            ch_types=["stim"],
        )
        stim_cnt = mne.io.RawArray(stim_chan[None], stim_info, verbose="WARNING")
        cnt = cnt.add_channels([stim_cnt])

        with h5py.File(self.filename, "r") as h5file:
            y = h5file["mrk"]["y"][:]
            y[np.sum(y, axis=1) == 0, -1] = 1
            event_i_classes = np.argmax(y, axis=1)

        if all_class_names == ["Right Hand", "Left Hand", "Rest", "Feet"]:
            durations = np.full(event_times_in_ms.shape, 4.0)
        else:
            durations = np.zeros_like(event_times_in_ms, dtype=float)

        descriptions = [all_class_names[i] for i in event_i_classes]
        annots = mne.Annotations(
            onset=event_times_in_ms / 1000.0,
            duration=durations,
            description=descriptions,
        )
        cnt.set_annotations(annots)
        return cnt
