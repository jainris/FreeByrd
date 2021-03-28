import sys
import os
import numpy as np
from ML.Utils.get_timestamps import get_timestamps
from ML.BirdVoxDetect import run_birdvoxdetect_segmentation
from ML.Clusterer.dominantset import get_clusters_using_dominant_sets
from ML.Clusterer.somandkmeans import get_clusters_using_SOM_and_k_means
from ML.FeatureExtraction.classification import get_activations
from ML.output import create_output


def main():
    # Takes arguments of form:
    # [output_path] [outdir] [clustering strategy (0 for dominant sets, 1 for SOM and K Means)] [threshold]
    # [noise_reduction (0/1)] [noise_file if noise_reduction == 1] [input_file_1] [input_file_2] ..... [input_file_n]
    # TODO: Better input parsing
    args = sys.argv[1:]
    output_path = args[0]
    outDir = args[1]
    dominant_set_clustering = int(args[2]) == 0
    threshold = float(args[3])
    noise_reduction = int(args[4]) == 1
    i = 5
    noise_file = None
    if noise_reduction:
        i = 6
        noise_file = args[5]
    file_names = args[i:]

    # Applying birdvoxdetect
    list_of_filenames_and_timestamps = run_birdvoxdetect_segmentation(
        file_names=file_names,
        noise_reduction=noise_reduction,
        noise_file=noise_file,
        threshold=threshold,
    )
    detection_map, birdnet_input = get_timestamps(list_of_filenames_and_timestamps)

    # Feature Extraction
    list_of_filenames_timestamps_activations = get_activations(birdnet_input)

    # Unzipping the list
    file_names, timestamps, features = tuple(
        zip(*list_of_filenames_timestamps_activations)
    )
    file_names, timestamps, features = (
        list(file_names),
        list(timestamps),
        np.array(features),
    )

    # Clustering
    if dominant_set_clustering:
        cluster_ids = get_clusters_using_dominant_sets(features)
    else:
        # Using SOM and K Means clustering.
        # Assuming the max number of clusters to be 9 times the number of files.
        cluster_ids = get_clusters_using_SOM_and_k_means(
            features,
            max_num_of_clusters=min(
                9 * len(list_of_filenames_and_timestamps), features.shape[0]
            ),
        )

    # Time to prepare the output!!!
    create_output(
        filenames=file_names,
        clusters=cluster_ids,
        timestamps=timestamps,
        output_path=output_path,
        detection_map=detection_map,
        outDir=outDir,
    )


if __name__ == "__main__":
    main()
