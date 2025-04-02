#!/usr/bin/env python3
"""
Enhanced Aerials Downloader

A tool to download Apple TV aerial screensavers from macOS.
Features:
- Download videos in different quality variants
- Filter by category and subcategory
- Download preview images
- Parallel downloads with configurable thread count
"""

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import requests
import tqdm
import urllib3
from iterfzf import iterfzf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aerials-downloader")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

QUALITY_VARIANTS = ["4KSDR240FPS", "4KSDR", "4KHDR", "2KSDR", "2KHDR", "2KAVC"]

QUALITY_URL_KEYS = {
    "4KSDR240FPS": "url-4K-SDR-240FPS",
    "4KSDR": "url-4K-SDR",
    "4KHDR": "url-4K-HDR",
    "2KSDR": "url-2K-SDR",
    "2KHDR": "url-2K-HDR",
    "2KAVC": "url-2K-AVC",
}


class AerialsDownloader:
    """Handles downloading and management of Apple TV aerial screensavers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the downloader with configuration.

        Args:
            config: A dictionary containing configuration parameters.
        """
        self.config = config
        self.base_path = Path(
            config.get(
                "base_path", "/Library/Application Support/com.apple.idleassetsd"
            )
        )
        self.customer_path = self.base_path / "Customer"
        self.json_file_path = self.customer_path / "entries.json"

        self.quality_variant = config.get("quality_variant", "4KSDR240FPS")
        if self.quality_variant not in QUALITY_VARIANTS:
            logger.warning(
                f"Unknown quality variant: {self.quality_variant}, falling back to 4KSDR240FPS"
            )
            self.quality_variant = "4KSDR240FPS"

        self.aerial_folder_path = self.customer_path / self.quality_variant
        self.sqlite_db_path = self.base_path / "Aerial.sqlite"

        self.snapshots_path = self.base_path / "snapshots"

        self.aerial_folder_path.mkdir(parents=True, exist_ok=True)

        self.download_previews = config.get("download_previews", False)
        if self.download_previews:
            self.preview_output_path = Path(
                config.get(
                    "preview_output_path", str(self.aerial_folder_path) + "_previews"
                )
            )
            self.preview_output_path.mkdir(parents=True, exist_ok=True)

        self.download_threads = config.get("download_threads", 1)
        self.max_retry = config.get("max_retry", 5)
        self.backoff_factor = config.get("backoff_factor", 1.5)
        self.chunk_size = config.get("chunk_size", 32) * 1024

    def load_aerials(self) -> List[Dict[str, Any]]:
        """
        Load aerial video data from the JSON file.

        Returns:
            A list of aerial video dictionaries.

        Raises:
            FileNotFoundError: If the JSON file cannot be found.
            json.JSONDecodeError: If the JSON file is invalid.
        """
        try:
            with open(self.json_file_path) as f:
                data = json.load(f)
                return data["assets"]
        except FileNotFoundError:
            logger.error(f"JSON file not found: {self.json_file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON file: {self.json_file_path}")
            raise

    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get the list of available categories.

        Returns:
            A list of category dictionaries.
        """
        try:
            with open(self.json_file_path) as f:
                data = json.load(f)
                return data["categories"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error getting categories: {str(e)}")
            return []

    def download_file(
        self, url: str, file_path: Path, name: str, resume_pos: int = 0
    ) -> bool:
        """
        Download a file with progress bar.

        Args:
            url: The URL to download from.
            file_path: The destination file path.
            name: A display name for the progress bar.
            resume_pos: Position to resume download from if interrupted.

        Returns:
            True if download was successful, False otherwise.
        """
        try:
            r = requests.head(url, verify=False, timeout=10)
            total = int(r.headers.get("content-length", 0))

            if (
                resume_pos == 0
                and file_path.exists()
                and file_path.stat().st_size == total
            ):
                logger.info(f"Already downloaded: {name}")
                return True

            with requests.get(
                url,
                stream=True,
                headers={"Range": f"bytes={resume_pos}-"},
                verify=False,
                timeout=30,
            ) as r:
                r.raise_for_status()

                mode = "wb" if resume_pos == 0 else "ab"
                with open(file_path, mode) as f:
                    with tqdm.tqdm(
                        desc=name,
                        total=total,
                        miniters=1,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        initial=resume_pos,
                    ) as pb:
                        for chunk in r.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pb.update(len(chunk))

            return True

        except Exception as e:
            logger.error(f"Error downloading {name}: {str(e)}")
            return False

    def download_aerial_with_retry(self, aerial: Dict[str, Any]) -> bool:
        """
        Download an aerial video with retry capability.

        Args:
            aerial: The aerial dictionary containing metadata.

        Returns:
            True if download was successful, False otherwise.
        """
        url_key = QUALITY_URL_KEYS.get(self.quality_variant)
        if not url_key or url_key not in aerial:
            logger.warning(
                f"No {self.quality_variant} URL for {aerial.get('id', 'unknown')}"
            )
            return False

        url = aerial[url_key].replace("\\", "")
        file_path = self.aerial_folder_path / f"{aerial['id']}.mov"
        temp_path = file_path.with_suffix(".mov.downloading")

        if file_path.exists() and self.is_file_complete(file_path, url):
            logger.info(f"Already downloaded: {aerial['accessibilityLabel']}")

            if self.download_previews:
                self.download_preview_for_aerial(aerial)

            return True

        retry = 0
        while retry < self.max_retry:
            try:
                resume_pos = temp_path.stat().st_size if temp_path.exists() else 0

                success = self.download_file(
                    url,
                    temp_path,
                    f"{aerial['accessibilityLabel']} [{retry+1}/{self.max_retry}]",
                    resume_pos=resume_pos,
                )

                if success:
                    temp_path.rename(file_path)

                    if self.download_previews:
                        self.download_preview_for_aerial(aerial)

                    return True

            except (
                requests.exceptions.ChunkedEncodingError,
                urllib3.exceptions.ProtocolError,
            ) as e:
                retry += 1
                sleep_time = self.backoff_factor * (2 ** (retry - 1))
                logger.warning(
                    f"Network error downloading {aerial['accessibilityLabel']} "
                    f"(retry {retry}/{self.max_retry}): {str(e)}. "
                    f"Retrying in {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(
                    f"Error downloading {aerial['accessibilityLabel']}: {str(e)}"
                )
                return False

        logger.error(
            f"Failed to download {aerial['accessibilityLabel']} after {self.max_retry} retries"
        )
        return False

    def download_preview_for_aerial(self, aerial: Dict[str, Any]) -> bool:
        """
        Download the preview image for an aerial video.

        Args:
            aerial: The aerial dictionary containing metadata.

        Returns:
            True if preview was successfully downloaded, False otherwise.
        """
        if not self.download_previews:
            return False

        aerial_id = aerial.get("id")
        if not aerial_id:
            return False

        preview_found = False

        try:
            for file_path in self.snapshots_path.glob(f"asset-preview-*.jpg"):
                dest_path = self.preview_output_path / f"{aerial_id}.jpg"
                if dest_path.exists():
                    preview_found = True
                    break

                shutil.copy(file_path, dest_path)
                logger.info(f"Downloaded preview for {aerial['accessibilityLabel']}")
                preview_found = True
                break

            return preview_found
        except Exception as e:
            logger.error(f"Error downloading preview for {aerial_id}: {str(e)}")
            return False

    def is_file_complete(self, file_path: Path, url: str) -> bool:
        """
        Check if a file is completely downloaded.

        Args:
            file_path: The path to the local file.
            url: The URL where the file is hosted.

        Returns:
            True if the file is complete, False otherwise.
        """
        try:
            local_size = file_path.stat().st_size
            r = requests.head(url, verify=False, timeout=10)
            remote_size = int(r.headers.get("content-length", 0))
            return local_size == remote_size
        except Exception as e:
            logger.error(f"Error checking file completeness: {str(e)}")
            return False

    def update_sqlite_db(self) -> bool:
        """
        Update the SQLite database to mark aerials as downloaded.

        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            con = sqlite3.connect(self.sqlite_db_path)
            cur = con.cursor()
            cur.execute("VACUUM;")
            cur.execute("UPDATE ZASSET SET ZLASTDOWNLOADED = 718364962.0204;")
            con.commit()
            con.close()
            logger.info("Updated Aerials database")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating SQLite database: {str(e)}")
            return False

    def restart_service(self) -> bool:
        """
        Restart the idleassetsd service.

        Returns:
            True if the service was restarted, False otherwise.
        """
        try:
            subprocess.run(["killall", "idleassetsd"], check=True)
            logger.info("Restarted idleassetsd service")
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Error restarting service: {str(e)}")
            return False

    def filter_by_category(
        self,
        aerials: List[Dict[str, Any]],
        category_id: Optional[str] = None,
        subcategory_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter aerials by category and subcategory.

        Args:
            aerials: The list of aerials to filter.
            category_id: Optional category ID to filter by.
            subcategory_id: Optional subcategory ID to filter by.

        Returns:
            A filtered list of aerials.
        """
        if not category_id:
            return aerials

        filtered_aerials = []
        aerials_set: Set[str] = set()

        for aerial in aerials:
            if subcategory_id:
                if any(
                    sub == subcategory_id for sub in aerial.get("subcategories", [])
                ):
                    if aerial["id"] not in aerials_set:
                        aerials_set.add(aerial["id"])
                        filtered_aerials.append(aerial)
            elif any(cat == category_id for cat in aerial.get("categories", [])):
                if aerial["id"] not in aerials_set:
                    aerials_set.add(aerial["id"])
                    filtered_aerials.append(aerial)

        return filtered_aerials

    def download_aerials(self, aerials: List[Dict[str, Any]]) -> None:
        """
        Download a list of aerials in parallel.

        Args:
            aerials: The list of aerials to download.
        """
        logger.info(
            f"Downloading {len(aerials)} aerials with {self.download_threads} threads"
        )

        quality_url_key = QUALITY_URL_KEYS.get(self.quality_variant)
        available_aerials = [a for a in aerials if quality_url_key in a]

        if len(available_aerials) < len(aerials):
            logger.warning(
                f"{len(aerials) - len(available_aerials)} aerials don't have the "
                f"selected quality variant ({self.quality_variant})"
            )

        with ThreadPoolExecutor(max_workers=self.download_threads) as executor:
            results = list(
                executor.map(self.download_aerial_with_retry, available_aerials)
            )

        successful = results.count(True)
        logger.info(
            f"Download complete: {successful}/{len(available_aerials)} successful"
        )

    def interactive_category_selection(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Interactively select a category and subcategory.

        Returns:
            A tuple containing the selected category ID and subcategory ID (both optional).
        """
        categories = self.get_categories()
        if not categories:
            logger.error("No categories found")
            return None, None

        print("\nSelect aerial category:")
        for i, category in enumerate(categories, 1):
            name = category["localizedNameKey"].replace("AerialCategory", "")
            print(f"{i}. {name}")
        print(f"{len(categories) + 1}. All")

        try:
            choice = int(input("\nEnter category number: "))
            if choice > len(categories) + 1 or choice < 1:
                logger.error("Invalid category selection")
                return None, None

            if choice == len(categories) + 1:
                return None, None

            selected_category = categories[choice - 1]
            category_id = selected_category["id"]

            subcategories = selected_category.get("subcategories", [])
            if subcategories:
                print("\nSelect subcategory:")
                for i, subcategory in enumerate(subcategories, 1):
                    name = subcategory["localizedNameKey"].replace(
                        "AerialSubcategory", ""
                    )
                    print(f"{i}. {name}")
                print(f"{len(subcategories) + 1}. All")

                subcategory_choice = int(input("\nEnter subcategory number: "))
                if (
                    subcategory_choice > len(subcategories) + 1
                    or subcategory_choice < 1
                ):
                    logger.error("Invalid subcategory selection")
                    return category_id, None

                if subcategory_choice == len(subcategories) + 1:
                    return category_id, None

                subcategory_id = subcategories[subcategory_choice - 1]["id"]
                return category_id, subcategory_id

            return category_id, None

        except (ValueError, IndexError) as e:
            logger.error(f"Error during selection: {str(e)}")
            return None, None

    def interactive_quality_selection(self) -> str:
        """
        Interactively select a quality variant.

        Returns:
            The selected quality variant.
        """
        print("\nSelect quality variant:")

        for i, quality in enumerate(QUALITY_VARIANTS, 1):
            info = ""
            if quality == "4KSDR240FPS":
                info = "(Highest quality, large files)"
            elif quality.startswith("4K"):
                info = "(4K resolution)"
            elif quality.startswith("2K"):
                info = "(2K resolution)"

            print(f"{i}. {quality} {info}")

        try:
            choice = int(
                input("\nEnter quality number [default is 1 - 4KSDR240FPS]: ") or "1"
            )
            if choice < 1 or choice > len(QUALITY_VARIANTS):
                logger.warning(f"Invalid choice, using default 4KSDR240FPS")
                return "4KSDR240FPS"

            selected_quality = QUALITY_VARIANTS[choice - 1]

            self.quality_variant = selected_quality
            self.aerial_folder_path = self.customer_path / selected_quality
            self.aerial_folder_path.mkdir(parents=True, exist_ok=True)

            return selected_quality

        except ValueError as e:
            logger.error(f"Error during quality selection: {str(e)}")
            return "4KSDR240FPS"

    def interactive_aerial_selection(
        self, filtered_aerials: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Allow the user to select specific aerials using fuzzy finding.

        Args:
            filtered_aerials: A pre-filtered list of aerials to choose from.

        Returns:
            A list of selected aerials.
        """
        if not filtered_aerials:
            logger.warning("No aerials to select from")
            return []

        name_mapping = {}

        def aerial_name(aerial: Dict[str, Any]) -> str:
            """Generate a display name for an aerial."""
            name = (
                f"{aerial['accessibilityLabel']} ({aerial.get('localizedNameKey', '')})"
            )
            name_mapping[name] = aerial
            return name

        def aerial_generator() -> Iterator[str]:
            for aerial in filtered_aerials:
                yield aerial_name(aerial)

        print(
            "\nUse fuzzy finder to select aerials (press TAB to select, ENTER when done):"
        )
        selected_names = iterfzf(
            aerial_generator(),
            multi=True,
        )

        if not selected_names:
            logger.warning("No aerials selected")
            return []

        selected_aerials = [
            name_mapping[name] for name in selected_names if name in name_mapping
        ]
        logger.info(f"Selected {len(selected_aerials)} aerials")
        return selected_aerials

    def run_interactive(self) -> None:
        """Run the downloader in interactive mode."""
        print("Welcome to Enhanced Aerials Downloader!")

        try:
            aerials = self.load_aerials()
            logger.info(f"Loaded {len(aerials)} aerial videos")

            self.quality_variant = self.interactive_quality_selection()
            logger.info(f"Selected quality variant: {self.quality_variant}")

            print("\nSelect an option:")
            print("1. Choose aerials manually")
            print("2. Download all aerials")
            print("3. Download previews only (no videos)")

            choice = input("\nEnter option number: ")

            if choice == "3":
                self.download_previews = True
                logger.info("Downloading previews only")
                for aerial in aerials:
                    self.download_preview_for_aerial(aerial)
            elif choice == "2":
                download_previews = (
                    input("\nDownload preview images? (y/n) [default: n]: ").lower()
                    == "y"
                )
                self.download_previews = download_previews
                self.download_aerials(aerials)
            elif choice == "1":
                category_id, subcategory_id = self.interactive_category_selection()

                filtered_aerials = self.filter_by_category(
                    aerials, category_id, subcategory_id
                )
                logger.info(f"Filtered to {len(filtered_aerials)} aerials")

                selected_aerials = self.interactive_aerial_selection(filtered_aerials)

                download_previews = (
                    input("\nDownload preview images? (y/n) [default: n]: ").lower()
                    == "y"
                )
                self.download_previews = download_previews

                if selected_aerials:
                    self.download_aerials(selected_aerials)
            else:
                logger.error("Invalid option")
                return

            self.update_sqlite_db()
            self.restart_service()

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise

    def run_batch(
        self, category: Optional[str] = None, subcategory: Optional[str] = None
    ) -> None:
        """
        Run in batch mode with optional category filtering.

        Args:
            category: Optional category name to filter by.
            subcategory: Optional subcategory name to filter by.
        """
        try:
            aerials = self.load_aerials()
            logger.info(f"Loaded {len(aerials)} aerial videos")

            category_id = None
            subcategory_id = None

            if category:
                categories = self.get_categories()
                for cat in categories:
                    if (
                        cat["localizedNameKey"].replace("AerialCategory", "")
                        == category
                    ):
                        category_id = cat["id"]

                        if subcategory:
                            for subcat in cat.get("subcategories", []):
                                if (
                                    subcat["localizedNameKey"].replace(
                                        "AerialSubcategory", ""
                                    )
                                    == subcategory
                                ):
                                    subcategory_id = subcat["id"]

            filtered_aerials = self.filter_by_category(
                aerials, category_id, subcategory_id
            )
            logger.info(f"Filtered to {len(filtered_aerials)} aerials")

            self.download_aerials(filtered_aerials)

            self.update_sqlite_db()
            self.restart_service()

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise


def check_root_privileges() -> bool:
    """
    Check if the script is running with root privileges.

    Returns:
        True if running as root, False otherwise.
    """
    return os.geteuid() == 0 if hasattr(os, "geteuid") else False


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced tool to download Apple TV aerial screensavers from macOS."
    )

    parser.add_argument(
        "--batch", action="store_true", help="Run in batch mode (non-interactive)"
    )

    parser.add_argument("--category", help="Filter by category (only in batch mode)")

    parser.add_argument(
        "--subcategory",
        help="Filter by subcategory (only in batch mode, requires --category)",
    )

    parser.add_argument(
        "--quality",
        choices=QUALITY_VARIANTS,
        default="4KSDR240FPS",
        help="Quality variant to download (default: 4KSDR240FPS)",
    )

    parser.add_argument(
        "--previews",
        action="store_true",
        help="Download preview images in addition to videos",
    )

    parser.add_argument(
        "--previews-only",
        action="store_true",
        help="Download only preview images, no videos",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("DOWNLOAD_THREADS", 1)),
        help="Number of parallel download threads",
    )

    parser.add_argument("--output", help="Custom output directory for downloaded files")

    parser.add_argument(
        "--preview-output", help="Custom output directory for preview images"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the application.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not check_root_privileges():
        logger.error("This script requires root privileges to access system files.")
        logger.error("Please run with sudo: sudo python3 app.py")
        return 1

    base_path = "/Library/Application Support/com.apple.idleassetsd"

    config = {
        "base_path": base_path,
        "quality_variant": args.quality,
        "download_threads": args.threads,
        "max_retry": 5,
        "backoff_factor": 1.5,
        "chunk_size": 32,
        "download_previews": args.previews or args.previews_only,
    }

    if args.output:
        config["aerial_folder_path"] = args.output

    if args.preview_output:
        config["preview_output_path"] = args.preview_output

    downloader = AerialsDownloader(config)

    try:
        if args.previews_only:
            aerials = downloader.load_aerials()
            logger.info(f"Loaded {len(aerials)} aerial videos")
            for aerial in aerials:
                downloader.download_preview_for_aerial(aerial)
            return 0
        elif args.batch:
            downloader.run_batch(category=args.category, subcategory=args.subcategory)
        else:
            downloader.run_interactive()
        return 0
    except Exception as e:
        logger.error(f"Failed to complete: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
