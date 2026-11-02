from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import shift
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .utils import dct_highpass


@dataclass
class BeatPartition:
    """Ordered beat groups with windows aligned to each group's beat indices."""

    groups: list[np.ndarray]
    group_windows: list[np.ndarray]
    ref_group_index: int | None
    ref_beat_index: int | None = None


@dataclass
class PartitionStep:
    """Ordered partitions and the update settings shared by one REACT step."""

    partitions: list[BeatPartition]
    take_mean_partitions: bool = True
    freeze_reference_group: bool = True


class BaseBeatPartitioner(ABC):
    """Group beats for REACT updates.

    ``param`` sets the iteration schedule: ``pre_group_niter``,
    ``group_niter``, and ``post_group_niter`` (defaults: 0, 1, 0).
    ``post_group_atomic=True`` ends with individual-beat updates.
    ``take_mean_partitions`` averages updates across partitions;
    ``freeze_reference_group`` skips the reference group's update.
    Both default to True. Groups without a reference are not frozen.
    """

    def __init__(self, *, param=None):
        param = dict(param or {})
        self.pre_group_niter = int(param.get("pre_group_niter", 0))
        self.group_niter = int(param.get("group_niter", 1))
        self.post_group_niter = int(param.get("post_group_niter", 0))
        self.post_group_atomic = bool(param.get("post_group_atomic", False))
        self.take_mean_partitions = bool(param.get("take_mean_partitions", True))
        self.freeze_reference_group = bool(param.get("freeze_reference_group", True))

    @property
    def inner_niter(self):
        """Number of partitioning steps in one outer loop."""
        return self.pre_group_niter + self.group_niter + self.post_group_niter

    @abstractmethod
    def make_partitions(self, trans, rotvecs, *, ind_iter, n_beat, rng):
        """Return a PartitionStep for one REACT iteration."""
        ...

    def _make_step(self, partitions):
        """Attach this partitioner's update settings to its ordered partitions."""
        return PartitionStep(
            partitions=partitions,
            take_mean_partitions=self.take_mean_partitions,
            freeze_reference_group=self.freeze_reference_group,
        )


class BaseFeatureExtractor(ABC):
    """Interface for constructing per-beat partitioning features."""

    @abstractmethod
    def extract_features(self, trans, rotvecs):
        """Return a feature matrix or ``None`` when features are unavailable."""
        ...


class KMeansBeatPartitioner(BaseBeatPartitioner, ABC):
    """Group beats by feature similarity using KMeans.

    ``feature_extractor`` supplies one feature row per beat. ``param``
    accepts the common BaseBeatPartitioner settings plus clustering
    options such as ``initial_n_clusters`` and ``n_clusters_offsets``.
    ``ordering`` is "distance" or "random". ``reference_selection`` is
    "tracked_compact" or "min_centroid_norm".
    """

    def __init__(self, feature_extractor, *, param=None):
        param = dict(param or {})
        super().__init__(param=param)
        self.feature_extractor = feature_extractor
        self.feature_source = str(param.get("feature_source", "motion"))
        self.n_init_search = int(param.get("n_init_search", 5))
        self.n_init_final = int(param.get("n_init_final", 20))
        self.initial_n_clusters = int(param.get("initial_n_clusters", 2))
        self.n_clusters_offsets = tuple(param.get("n_clusters_offsets", (0,)))
        self.ordering = str(param.get("ordering", "distance"))
        self.reference_selection = str(
            param.get("reference_selection", "tracked_compact")
        )
        if self.feature_source not in (
            "inav",
            "motion",
            "delta",
            "motion_and_delta",
            "extern",
        ):
            raise ValueError(
                "feature_source must be 'inav', 'motion', 'delta', "
                f"'motion_and_delta', or 'extern', got {self.feature_source!r}."
            )
        if self.initial_n_clusters < 2:
            raise ValueError(
                "initial_n_clusters must be at least 2, "
                f"got {self.initial_n_clusters}."
            )
        if self.ordering not in ("distance", "random"):
            raise ValueError(
                "ordering must be either 'distance' or 'random', "
                f"got {self.ordering!r}."
            )
        if self.reference_selection not in (
            "tracked_compact",
            "min_centroid_norm",
        ):
            raise ValueError(
                "reference_selection must be either 'tracked_compact' or "
                f"'min_centroid_norm', got {self.reference_selection!r}."
            )

        self._reset_state()

    def _reset_state(self):
        self.reference_beat = None
        self._previous_groups = None

    def _clustering_is_active(self, ind_iter):
        """Return whether fresh clustering should run in this iteration."""
        return ind_iter + 1 <= self.group_niter

    def make_partitions(self, trans, rotvecs, *, ind_iter, n_beat, rng):
        """Return beat groups for the current iteration.

        Recluster during the grouping phase. Afterwards, reuse the last groups
        or switch to individual beats according to ``post_group_atomic``.
        """

        ind_iter = int(ind_iter)
        n_beat = int(n_beat)

        if ind_iter == 0:
            self._reset_state()

        if ind_iter < self.pre_group_niter:
            groups = [
                np.asarray([index], dtype=int) for index in rng.permutation(n_beat)
            ]
            return self._make_step([self._make_partition(groups, None)])

        group_iter = ind_iter - self.pre_group_niter
        clustering_active = self._clustering_is_active(group_iter)

        if not clustering_active and not self.post_group_atomic:
            groups = self._shuffled_previous_groups(rng)
            return self._make_step([self._make_partition(groups, self.reference_beat)])

        features = None
        if clustering_active:
            features = self.feature_extractor.extract_features(trans, rotvecs)

        if features is None:
            groups, self.reference_beat = self._singleton_groups(
                n_beat, rng, self.reference_beat
            )
            return self._make_step([self._store_partition(groups, self.reference_beat)])

        features = np.asarray(features)
        cluster_labels, n_clusters = self._cluster_labels(features, group_iter, rng)
        groups, central_reference_beat = self._ordered_groups(
            features,
            cluster_labels,
            rng,
            self.reference_beat,
        )
        self.reference_beat = central_reference_beat
        central_partition = self._store_partition(
            groups,
            central_reference_beat,
        )

        cluster_counts = sorted(
            {
                n_clusters + offset
                for offset in self.n_clusters_offsets
                if offset != 0 and 2 <= n_clusters + offset <= n_beat
            }
        )

        partitions = [central_partition]
        for cluster_count in cluster_counts:
            partition_labels = self._fit_kmeans(
                features,
                cluster_count,
                rng,
                self.n_init_final,
            )
            partition_groups, partition_reference_beat = self._ordered_groups(
                features,
                partition_labels,
                rng,
                central_reference_beat,
            )
            partitions.append(
                self._make_partition(
                    partition_groups,
                    partition_reference_beat,
                )
            )

        return self._make_step(partitions)

    @abstractmethod
    def _cluster_labels(self, features, ind_iter, rng):
        """Return central cluster labels and the requested cluster count."""
        ...

    @staticmethod
    def _fit_kmeans(features, n_clusters, rng, n_init):
        random_state = int(rng.integers(np.iinfo(np.int32).max))
        model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )
        return model.fit_predict(features)

    def _ordered_groups(self, features, cluster_labels, rng, reference_beat):
        cluster_ids = [int(cluster_id) for cluster_id in np.unique(cluster_labels)]
        centers = {
            cluster_id: np.mean(features[cluster_labels == cluster_id], axis=0)
            for cluster_id in cluster_ids
        }

        if self.reference_selection == "min_centroid_norm":
            reference_cluster = min(
                cluster_ids,
                key=lambda cluster_id: (
                    float(np.sum(centers[cluster_id] ** 2)),
                    cluster_id,
                ),
            )
        elif reference_beat is None:
            compactness = {}
            for cluster_id in cluster_ids:
                cluster_features = features[cluster_labels == cluster_id]
                sqdist = np.sum((cluster_features - centers[cluster_id]) ** 2, axis=1)
                compactness[cluster_id] = float(np.mean(sqdist))
            reference_cluster = min(
                cluster_ids,
                key=lambda cluster_id: (compactness[cluster_id], cluster_id),
            )
        else:
            reference_feature = features[int(reference_beat)]
            reference_cluster = min(
                cluster_ids,
                key=lambda cluster_id: (
                    float(np.sum((centers[cluster_id] - reference_feature) ** 2)),
                    cluster_id,
                ),
            )

        reference_indices = np.flatnonzero(cluster_labels == reference_cluster)
        reference_sqdist = np.sum(
            (features[reference_indices] - centers[reference_cluster]) ** 2,
            axis=1,
        )
        reference_beat = int(reference_indices[np.argmin(reference_sqdist)])

        nonreference_ids = [
            cluster_id for cluster_id in cluster_ids if cluster_id != reference_cluster
        ]
        if self.ordering == "distance":
            reference_center = centers[reference_cluster]
            nonreference_ids.sort(
                key=lambda cluster_id: (
                    -float(np.sum((centers[cluster_id] - reference_center) ** 2)),
                    cluster_id,
                )
            )
        else:
            permutation = rng.permutation(len(nonreference_ids))
            nonreference_ids = [nonreference_ids[idx] for idx in permutation]

        ordered_ids = nonreference_ids + [reference_cluster]
        groups = [
            np.flatnonzero(cluster_labels == cluster_id) for cluster_id in ordered_ids
        ]
        return groups, reference_beat

    @staticmethod
    def _singleton_groups(n_beat, rng, reference_beat):
        if reference_beat is None:
            ordered_beats = rng.permutation(n_beat).tolist()
            reference_beat = int(ordered_beats[-1])
        else:
            nonreference_beats = [
                index for index in range(n_beat) if index != reference_beat
            ]
            ordered_beats = rng.permutation(nonreference_beats).tolist()
            ordered_beats.append(reference_beat)

        groups = [np.asarray([index], dtype=int) for index in ordered_beats]
        return groups, reference_beat

    @staticmethod
    def _make_partition(groups, reference_beat):
        return BeatPartition(
            groups=groups,
            group_windows=[np.ones(len(group), dtype="float32") for group in groups],
            ref_group_index=(None if reference_beat is None else len(groups) - 1),
            ref_beat_index=reference_beat,
        )

    def _store_partition(self, groups, reference_beat):
        self._previous_groups = [
            np.asarray(beat_indices, dtype=int).copy() for beat_indices in groups
        ]
        groups = [beat_indices.copy() for beat_indices in self._previous_groups]
        return self._make_partition(groups, reference_beat)

    def _shuffled_previous_groups(self, rng):
        permutation = rng.permutation(len(self._previous_groups) - 1)
        groups = [self._previous_groups[index].copy() for index in permutation]
        groups.append(self._previous_groups[-1].copy())
        return groups


class ClusterCountKMeansPartitioner(KMeansBeatPartitioner):
    """Double the initial KMeans cluster count with each iteration."""

    def _cluster_labels(self, features, ind_iter, rng):
        n_beat = features.shape[0]
        n_clusters = min(self.initial_n_clusters * 2**ind_iter, n_beat)
        labels = self._fit_kmeans(
            features,
            n_clusters,
            rng,
            self.n_init_final,
        )
        return labels, n_clusters


class MaxGroupSizeKMeansPartitioner(KMeansBeatPartitioner):
    """Adapt the cluster count to reduce the largest cluster by half."""

    def _reset_state(self):
        super()._reset_state()
        self.previous_max_group_size = None
        self.previous_n_clusters = None

    def _cluster_labels(self, features, ind_iter, rng):
        n_beat = features.shape[0]
        if self.previous_max_group_size is None:
            requested_clusters = min(self.initial_n_clusters, n_beat)
            labels, max_group_size = self._fit_and_measure(
                features,
                requested_clusters,
                rng,
                self.n_init_final,
            )
            self._save_search_result(requested_clusters, max_group_size)
            return labels, requested_clusters

        target_max_group_size = max(1, self.previous_max_group_size // 2)
        first_n_clusters = max(
            self.previous_n_clusters + 1,
            int(np.ceil(n_beat / target_max_group_size)),
        )
        first_n_clusters = min(first_n_clusters, n_beat)

        best_search_max_group_size = n_beat + 1
        best_search_n_clusters = None
        final_results = {}

        for n_clusters in range(first_n_clusters, n_beat + 1):
            _, search_max_group_size = self._fit_and_measure(
                features,
                n_clusters,
                rng,
                self.n_init_search,
            )
            if search_max_group_size < best_search_max_group_size:
                best_search_max_group_size = search_max_group_size
                best_search_n_clusters = n_clusters

            if search_max_group_size > target_max_group_size:
                continue

            final_labels, final_max_group_size = self._fit_and_measure(
                features,
                n_clusters,
                rng,
                self.n_init_final,
            )
            final_results[n_clusters] = (final_labels, final_max_group_size)
            if final_max_group_size <= target_max_group_size:
                self._save_search_result(n_clusters, final_max_group_size)
                return final_labels, n_clusters

        if best_search_n_clusters not in final_results:
            final_labels, final_max_group_size = self._fit_and_measure(
                features,
                best_search_n_clusters,
                rng,
                self.n_init_final,
            )
            final_results[best_search_n_clusters] = (
                final_labels,
                final_max_group_size,
            )

        best_n_clusters, (best_labels, best_max_group_size) = min(
            final_results.items(),
            key=lambda item: (item[1][1], item[0]),
        )

        self._save_search_result(best_n_clusters, best_max_group_size)
        return best_labels, best_n_clusters

    def _fit_and_measure(self, features, n_clusters, rng, n_init):
        labels = self._fit_kmeans(features, n_clusters, rng, n_init)
        return labels, self._largest_cluster_size(labels)

    @staticmethod
    def _largest_cluster_size(labels):
        _, counts = np.unique(labels, return_counts=True)
        return int(np.max(counts))

    def _save_search_result(self, n_clusters, max_group_size):
        self.previous_n_clusters = int(n_clusters)
        self.previous_max_group_size = int(max_group_size)


class SubatomicKMeansBeatPartitioner(BaseBeatPartitioner):
    """Schedule heartbeat and within-heartbeat motion updates.

    ``param`` contains ``n_trs_per_beat`` (default 18) and four stage
    dictionaries, run in order: ``atomic_kmeans``, ``atomic``,
    ``subatomic_kmeans``, and ``subatomic``. Each stage's ``niter`` defaults
    to zero, which skips that stage.

    The KMeans stages group similar heartbeats. Feature settings come from
    ``atomic_kmeans``. Both within-heartbeat stages use ``subatomic`` window
    settings: ``plateau_length``, ``ramp_length``, and ``window_type``.
    KMeans stages also accept ``take_mean_partitions`` and
    ``freeze_reference_group``; freezing does not prevent overlapping
    windows from changing the same TRs.

    Group indices refer to flattened heartbeat/TR data. Use with default
    REACT only; total iterations must equal the sum of all stage counts.
    """

    def __init__(self, feature_extractor, *, param=None):
        param = dict(param or {})
        super().__init__(param=param)
        atomic_kmeans_param = dict(param.get("atomic_kmeans") or {})
        atomic_param = dict(param.get("atomic") or {})
        subatomic_kmeans_param = dict(param.get("subatomic_kmeans") or {})
        subatomic_param = dict(param.get("subatomic") or {})
        self.n_trs_per_beat = int(param.get("n_trs_per_beat", 18))
        self.atomic_kmeans_niter = int(atomic_kmeans_param.pop("niter", 0))
        self.atomic_niter = int(atomic_param.get("niter", 0))
        self.subatomic_kmeans_niter = int(subatomic_kmeans_param.pop("niter", 0))
        self.subatomic_niter = int(subatomic_param.get("niter", 0))
        self.plateau_length = int(subatomic_param.get("plateau_length", 6))
        self.ramp_length = int(subatomic_param.get("ramp_length", 3))
        self.window_type = str(subatomic_param.get("window_type", "trapezoid"))

        atomic_kmeans_param.update(
            pre_group_niter=0,
            group_niter=self.atomic_kmeans_niter,
            post_group_niter=0,
            post_group_atomic=False,
        )
        self.atomic_kmeans_partitioner = MaxGroupSizeKMeansPartitioner(
            feature_extractor, param=atomic_kmeans_param
        )
        self.feature_source = self.atomic_kmeans_partitioner.feature_source
        subatomic_kmeans_param.update(
            feature_source=self.feature_source,
            pre_group_niter=0,
            group_niter=self.subatomic_kmeans_niter,
            post_group_niter=0,
            post_group_atomic=False,
        )
        self.subatomic_kmeans_partitioner = MaxGroupSizeKMeansPartitioner(
            feature_extractor=feature_extractor,
            param=subatomic_kmeans_param,
        )

        window_builder = ContiguousOverlappingBeatPartitioner(
            param={"window_type": self.window_type}
        )
        boundaries = np.append(
            np.arange(0, self.n_trs_per_beat, self.plateau_length, dtype=int),
            self.n_trs_per_beat,
        )
        template = window_builder._make_partition(
            boundaries, window_builder._make_ramp(self.ramp_length), rng=None
        )
        self._subatomic_groups = template.groups
        self._subatomic_windows = template.group_windows
        plateau_centers = np.asarray(
            [
                group[window == 1].mean()
                for group, window in zip(
                    self._subatomic_groups, self._subatomic_windows
                )
            ]
        )
        self._reference_window_index = int(
            np.argmin(np.abs(plateau_centers - (self.n_trs_per_beat - 1) / 2))
        )
        self._reference_tr_offset = int(
            np.floor(plateau_centers[self._reference_window_index])
        )
        self._reset_state()

    @property
    def inner_niter(self):
        """Total default-REACT iteration count across all four stages."""
        return (
            self.atomic_kmeans_niter
            + self.atomic_niter
            + self.subatomic_kmeans_niter
            + self.subatomic_niter
        )

    def _reset_state(self):
        """Reset both independent clustering schedules and atomic reference."""
        self.atomic_kmeans_partitioner._reset_state()
        self.subatomic_kmeans_partitioner._reset_state()
        self._atomic_reference_beat = None

    def _mean_heartbeat_motion(self, motion):
        """Average motion parameters over all TRs within each heartbeat."""
        motion = np.asarray(motion)
        return motion.reshape(-1, self.n_trs_per_beat, motion.shape[-1]).mean(axis=1)

    def make_partitions(self, trans, rotvecs, *, ind_iter, n_beat, rng):
        """Dispatch by stage, with local iteration numbering for fresh KMeans."""
        ind_iter = int(ind_iter)
        n_beat = int(n_beat)
        if n_beat < 1 or n_beat % self.n_trs_per_beat:
            raise ValueError(
                "The TR count must be a positive multiple of n_trs_per_beat."
            )
        if not 0 <= ind_iter < self.inner_niter:
            raise ValueError("ind_iter is outside the configured four-stage schedule.")
        if ind_iter == 0:
            self._reset_state()

        n_heartbeat = n_beat // self.n_trs_per_beat
        if ind_iter < self.atomic_kmeans_niter:
            step = self.atomic_kmeans_partitioner.make_partitions(
                self._mean_heartbeat_motion(trans),
                self._mean_heartbeat_motion(rotvecs),
                ind_iter=ind_iter,
                n_beat=n_heartbeat,
                rng=rng,
            )
            return PartitionStep(
                partitions=[self._expand_heartbeat_groups(p) for p in step.partitions],
                take_mean_partitions=step.take_mean_partitions,
                freeze_reference_group=step.freeze_reference_group,
            )

        subatomic_kmeans_start = self.atomic_kmeans_niter + self.atomic_niter
        if ind_iter < subatomic_kmeans_start:
            if ind_iter == self.atomic_kmeans_niter:
                self._atomic_reference_beat = (
                    self.atomic_kmeans_partitioner.reference_beat
                )
            groups, self._atomic_reference_beat = (
                KMeansBeatPartitioner._singleton_groups(
                    n_heartbeat, rng, self._atomic_reference_beat
                )
            )
            partition = KMeansBeatPartitioner._make_partition(
                groups, self._atomic_reference_beat
            )
            return PartitionStep(
                partitions=[self._expand_heartbeat_groups(partition)],
                take_mean_partitions=False,
                freeze_reference_group=False,
            )

        subatomic_start = subatomic_kmeans_start + self.subatomic_kmeans_niter
        if ind_iter < subatomic_start:
            step = self.subatomic_kmeans_partitioner.make_partitions(
                self._mean_heartbeat_motion(trans),
                self._mean_heartbeat_motion(rotvecs),
                ind_iter=ind_iter - subatomic_kmeans_start,
                n_beat=n_heartbeat,
                rng=rng,
            )
            return PartitionStep(
                partitions=[
                    self._expand_subatomic_clusters(partition, rng)
                    for partition in step.partitions
                ],
                take_mean_partitions=step.take_mean_partitions,
                freeze_reference_group=step.freeze_reference_group,
            )

        return self._make_subatomic_step(n_heartbeat, rng)

    def _expand_heartbeat_groups(self, partition):
        """Map heartbeat groups and weights to every TR of their members."""
        tr_offsets = np.arange(self.n_trs_per_beat, dtype=int)
        return BeatPartition(
            groups=[
                (group[:, None] * self.n_trs_per_beat + tr_offsets).ravel()
                for group in partition.groups
            ],
            group_windows=[
                np.repeat(window, self.n_trs_per_beat)
                for window in partition.group_windows
            ],
            ref_group_index=partition.ref_group_index,
            ref_beat_index=(
                None
                if partition.ref_beat_index is None
                else partition.ref_beat_index * self.n_trs_per_beat
            ),
        )

    def _expand_subatomic_clusters(self, partition, rng):
        """Expand ordered heartbeat clusters into shuffled window groups."""
        groups = []
        windows = []
        ref_group_index = None
        for cluster_index, heartbeats in enumerate(partition.groups):
            is_reference = cluster_index == partition.ref_group_index
            window_order = rng.permutation(len(self._subatomic_groups)).tolist()
            if is_reference:
                window_order.remove(self._reference_window_index)
                window_order.append(self._reference_window_index)
            for window_index in window_order:
                groups.append(
                    (
                        heartbeats[:, None] * self.n_trs_per_beat
                        + self._subatomic_groups[window_index]
                    ).ravel()
                )
                windows.append(
                    np.tile(self._subatomic_windows[window_index], len(heartbeats))
                )
                if is_reference and window_index == self._reference_window_index:
                    ref_group_index = len(groups) - 1

        return BeatPartition(
            groups=groups,
            group_windows=windows,
            ref_group_index=ref_group_index,
            ref_beat_index=(
                None
                if partition.ref_beat_index is None
                else partition.ref_beat_index * self.n_trs_per_beat
                + self._reference_tr_offset
            ),
        )

    def _make_subatomic_step(self, n_heartbeat, rng):
        """Visit each heartbeat/window pair once in randomized sequential order."""
        remaining = [
            list(range(len(self._subatomic_groups))) for _ in range(n_heartbeat)
        ]
        active_heartbeats = list(range(n_heartbeat))
        groups = []
        windows = []
        while active_heartbeats:
            heartbeat_position = int(rng.integers(len(active_heartbeats)))
            heartbeat = active_heartbeats[heartbeat_position]
            group_position = int(rng.integers(len(remaining[heartbeat])))
            group_index = remaining[heartbeat].pop(group_position)
            groups.append(
                self._subatomic_groups[group_index] + heartbeat * self.n_trs_per_beat
            )
            windows.append(self._subatomic_windows[group_index].copy())
            if not remaining[heartbeat]:
                active_heartbeats.pop(heartbeat_position)
        return PartitionStep(
            partitions=[BeatPartition(groups, windows, ref_group_index=None)],
            take_mean_partitions=False,
            freeze_reference_group=False,
        )


class DummyFeatureExtractor(BaseFeatureExtractor):
    """Returns precomputed features."""

    def __init__(self, features):
        self.features = features

    def extract_features(self, trans, rotvecs):
        return self.features


class MotionFeatureExtractor(BaseFeatureExtractor):
    """Construct grouping features from current rigid motion."""

    def __init__(self, *, param=None):
        param = dict(param or {})
        self.rot_to_cm = float(param.get("rot_to_cm", 10.0))
        self.cutoff = int(param.get("cutoff", 20))

    def extract_features(self, trans, rotvecs):
        trans = np.asarray(trans)
        rotvecs = np.asarray(rotvecs)

        feature_parts = []
        if np.any(trans):
            feature_parts.append(trans)
        if np.any(rotvecs):
            feature_parts.append(self.rot_to_cm * rotvecs)
        if len(feature_parts) == 0:
            return None

        raw_features = np.concatenate(feature_parts, axis=1).astype(float, copy=False)
        if self.cutoff <= 0:
            return raw_features

        features = np.empty_like(raw_features)
        for feature_index in range(raw_features.shape[1]):
            features[:, feature_index] = dct_highpass(
                raw_features[:, feature_index], cutoff=self.cutoff
            )
        return features


def extract_features_inav(*, pca_thresh, inavs, masks, spacings, planes, trans):
    """Return PCA features from translation-corrected iNAV images.

    ``inavs`` has shape (n_plane, n_beat, ...); provide one mask, pixel
    spacing, and plane label per plane. Labels are "cor", "sag", or "3d".
    Spacings follow image-axis order. Coronal axes are (-z, x), sagittal
    axes are (-z, y), and 3D axes are (z, y, x).
    ``trans`` has shape (n_beat, 3), in physical (x, y, z) units.

    ``pca_thresh`` is the retained variance fraction (0 < value < 1) or
    component count. The result has shape (n_beat, n_components).
    """
    n_plane, n_beat = inavs.shape[:2]

    # --- prepare image shift --- #
    # trans_px: (height, width) in pixels
    # spacings: (height, width) in physical units
    # trans: (x, y, z) in physical units, so they are (width, height, depth)

    trans_px = []
    for ind_p, plane in enumerate(planes):
        if plane == "cor":  # height -z, width x
            trans_px.append(
                np.stack(
                    [
                        -trans[:, 2] / spacings[ind_p][0],
                        trans[:, 0] / spacings[ind_p][1],
                    ],
                    axis=1,
                )
            )
        elif plane == "sag":  # height -z, width y
            trans_px.append(
                np.stack(
                    [
                        -trans[:, 2] / spacings[ind_p][0],
                        trans[:, 1] / spacings[ind_p][1],
                    ],
                    axis=1,
                )
            )
        elif plane == "3d":
            trans_px.append(np.asarray(trans)[:, ::-1] / spacings[ind_p])
        else:
            raise ValueError(f"Unknown plane: {plane}")

    # --- apply correction --- #

    corrected_inavs = np.zeros_like(inavs)
    for ind_p in range(n_plane):
        for ind_b in range(n_beat):
            corrected_inavs[ind_p, ind_b] = shift(
                inavs[ind_p, ind_b], trans_px[ind_p][ind_b]
            )

    # --- prepare features --- #

    masks_bool = [mask > 0 for mask in masks]
    raw_features = []
    for ind_b in range(n_beat):
        feats = []
        for ind_p in range(n_plane):
            data = corrected_inavs[ind_p][ind_b]
            mask = masks_bool[ind_p]

            mu = data[mask].mean()
            sigma = data[mask].std()
            normalized = (data - mu) / sigma
            feats.append(normalized[mask])

        raw_features.append(np.concatenate(feats))
    raw_features = np.stack(raw_features, axis=0)

    # --- PCA --- #

    pca = PCA(n_components=pca_thresh, svd_solver="full")
    return pca.fit_transform(raw_features)


class InavFeatureExtractor(BaseFeatureExtractor):
    """Construct motion-corrected iNAV features for the current iteration."""

    def __init__(
        self,
        inavs,
        inav_masks,
        inav_spacings,
        inav_planes,
        *,
        param=None,
    ):
        param = dict(param or {})
        self.inavs = np.asarray(inavs)
        self.inav_masks = inav_masks
        self.inav_spacings = inav_spacings
        self.inav_planes = inav_planes
        self.pca_thresh = param.get("pca_thresh", 0.9)

    def extract_features(self, trans, rotvecs):
        return extract_features_inav(
            pca_thresh=self.pca_thresh,
            inavs=self.inavs,
            masks=self.inav_masks,
            spacings=self.inav_spacings,
            planes=self.inav_planes,
            trans=trans,
        )


class ContiguousOverlappingBeatPartitioner(BaseBeatPartitioner):
    """Group neighboring beats with overlapping tapered windows.

    ``param`` accepts the common BaseBeatPartitioner settings.
    ``init_n_group`` sets the starting group count (default 2), which
    doubles during the grouping phase. ``atomic_threshold`` selects when
    to switch to individual beats (default 3).
    ``ramp_ratio`` controls overlap; zero disables it. ``window_type`` is
    "trapezoid" or "cosine". ``n_shift`` adds shifted partitions.
    Use ``post_group_atomic=True`` when ``group_niter=0``.
    """

    def __init__(self, *, param=None):
        param = dict(param or {})
        super().__init__(param=param)
        self.init_n_group = int(param.get("init_n_group", 2))
        self.atomic_threshold = int(param.get("atomic_threshold", 3))
        self.ramp_ratio = float(param.get("ramp_ratio", 1.0))
        self.window_type = str(param.get("window_type", "trapezoid"))
        self.n_shift = int(param.get("n_shift", 1))

        if self.init_n_group < 1:
            raise ValueError("init_n_group must be at least one.")
        if self.atomic_threshold < 1:
            raise ValueError("atomic_threshold must be at least one.")
        if not np.isfinite(self.ramp_ratio) or self.ramp_ratio < 0:
            raise ValueError("ramp_ratio must be finite and nonnegative.")
        if self.group_niter < 0:
            raise ValueError("group_niter must be nonnegative.")
        if self.group_niter == 0 and not self.post_group_atomic:
            raise ValueError("group_niter=0 requires post_group_atomic=True.")
        if self.window_type not in ("trapezoid", "cosine"):
            raise ValueError("window_type must be 'trapezoid' or 'cosine'.")
        if self.n_shift < 0:
            raise ValueError("n_shift must be nonnegative.")

    def _make_ramp(self, ramp_length):
        """Construct the rising ramp; the falling ramp is one minus this."""
        ramp = np.arange(1, ramp_length + 1, dtype="float64")
        ramp /= ramp_length + 1
        if self.window_type == "cosine":
            ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))
        return ramp.astype("float32")

    def make_partitions(self, trans, rotvecs, *, ind_iter, n_beat, rng):
        """Return shuffled window partitions or a scheduled atomic partition."""
        ind_iter = int(ind_iter)
        n_beat = int(n_beat)
        if n_beat < 1:
            raise ValueError("n_beat must be at least one.")

        if ind_iter < self.pre_group_niter:
            return self._make_step([self._make_atomic_partition(n_beat, rng)])
        group_iter = ind_iter - self.pre_group_niter
        if group_iter >= self.group_niter:
            if self.post_group_atomic:
                return self._make_step([self._make_atomic_partition(n_beat, rng)])
            group_iter = self.group_niter - 1

        n_group = min(self.init_n_group * 2**group_iter, n_beat)
        group_size, remainder = divmod(n_beat, n_group)
        group_sizes = np.full(n_group, group_size, dtype=int)
        group_sizes[:remainder] += 1
        if group_sizes.max() < self.atomic_threshold:
            return self._make_step([self._make_atomic_partition(n_beat, rng)])

        plateau_length = max(1, round(n_beat / n_group))
        ramp_length = round(self.ramp_ratio * plateau_length)
        ramp = self._make_ramp(ramp_length)
        boundaries = np.concatenate(([0], np.cumsum(group_sizes)))

        offset_period = int(group_sizes.max())
        base_offset = self._choose_base_offset(boundaries)
        offsets = sorted(
            {
                (base_offset + shift_index * offset_period // (self.n_shift + 1))
                % offset_period
                for shift_index in range(self.n_shift + 1)
            }
        )
        rng.shuffle(offsets)
        return self._make_step(
            [
                self._make_partition(
                    self._shift_boundaries(boundaries, offset), ramp, rng
                )
                for offset in offsets
            ]
        )

    @staticmethod
    def _shift_boundaries(boundaries, offset):
        """Shift plateau boundaries while keeping both acquisition ends fixed."""
        n_beat = int(boundaries[-1])
        shifted = boundaries[:-1] + offset
        interior = shifted[(shifted > 0) & (shifted < n_beat)]
        return np.concatenate(([0], interior, [n_beat]))

    def _choose_base_offset(self, boundaries):
        """Balance plateau sizes; retain the lowest offset on an exact tie."""
        offset_period = int(np.diff(boundaries).max())
        best_offset = 0
        best_score = None
        for offset in range(offset_period):
            group_sizes = np.diff(self._shift_boundaries(boundaries, offset))
            score = (int(group_sizes.min()), -float(group_sizes.var()))
            if best_score is None or score > best_score:
                best_offset = offset
                best_score = score
        return best_offset

    def _make_partition(self, boundaries, ramp, rng):
        """Attach and crop ramps; shuffle groups and windows when RNG is given."""
        n_beat = int(boundaries[-1])
        ramp_length = len(ramp)
        groups = []
        group_windows = []
        for plateau_start, plateau_stop in zip(boundaries[:-1], boundaries[1:]):
            window = np.concatenate(
                (
                    ramp,
                    np.ones(plateau_stop - plateau_start, dtype="float32"),
                    1.0 - ramp,
                )
            )
            window_start = plateau_start - ramp_length
            start = max(0, window_start)
            stop = min(n_beat, plateau_stop + ramp_length)
            groups.append(np.arange(start, stop, dtype=int))
            group_windows.append(
                window[start - window_start : stop - window_start].copy()
            )

        order = range(len(groups)) if rng is None else rng.permutation(len(groups))
        return BeatPartition(
            groups=[groups[index] for index in order],
            group_windows=[group_windows[index] for index in order],
            ref_group_index=None,
        )

    @staticmethod
    def _make_atomic_partition(n_beat, rng):
        groups = [np.asarray([index], dtype=int) for index in rng.permutation(n_beat)]
        return BeatPartition(
            groups=groups,
            group_windows=[np.ones(1, dtype="float32") for _ in groups],
            ref_group_index=None,
        )
