from evaluation import visualize_tsne_csv

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Visualize t-SNE CSV interactively"
    )

    parser.add_argument(
        "file",
        type=str,
        help="Path to the CSV file"
    )

    args = parser.parse_args()

    visualize_tsne_csv(args.file)


if __name__ == "__main__":
    main()