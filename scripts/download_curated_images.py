#!/usr/bin/env python3
"""
Curated Image Downloader for Fall Hazard Detection Research

Downloads open-source images from Unsplash organized by risk level and room type.
Creates a balanced dataset for comparing VLM performance on hazard detection.

Usage:
    # Download all images (requires Unsplash API key)
    python scripts/download_curated_images.py --api-key YOUR_UNSPLASH_KEY

    # Download specific category
    python scripts/download_curated_images.py --risk-level high --room bathroom

    # List URLs without downloading (no API key needed)
    python scripts/download_curated_images.py --list-only

Note: Unsplash free tier allows 50 requests/hour.
"""

import os
import sys
import json
import argparse
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Search queries organized by risk level and room type
SEARCH_QUERIES = {
    "low_risk": {
        "bathroom": [
            "modern bathroom accessible",
            "clean bathroom grab bars",
            "bright bathroom interior",
            "accessible bathroom design"
        ],
        "kitchen": [
            "clean organized kitchen",
            "modern kitchen interior",
            "bright kitchen design",
            "minimalist kitchen"
        ],
        "stairs": [
            "modern staircase handrail",
            "well lit stairs interior",
            "safe staircase design",
            "wooden stairs with railing"
        ],
        "living_room": [
            "organized living room",
            "clean home interior",
            "bright living room",
            "minimalist living room"
        ]
    },
    "high_risk": {
        "bathroom": [
            "old bathroom",
            "small bathroom cluttered",
            "vintage bathroom",
            "cramped bathroom"
        ],
        "kitchen": [
            "messy kitchen",
            "cluttered kitchen counter",
            "small kitchen crowded",
            "old kitchen interior"
        ],
        "stairs": [
            "old wooden stairs",
            "narrow staircase",
            "steep stairs",
            "dark stairway"
        ],
        "living_room": [
            "cluttered living room",
            "messy room interior",
            "crowded living space",
            "old living room"
        ]
    }
}

# Alternative: Direct Unsplash URLs (curated, no API needed)
CURATED_URLS = {
    "low_risk": {
        "bathroom": [
            "https://images.unsplash.com/photo-1552321554-5fefe8c9ef14",  # Modern bathroom
            "https://images.unsplash.com/photo-1620626011761-996317b8d101",  # Clean bathroom
            "https://images.unsplash.com/photo-1584622650111-993a426fbf0a",  # Bright bathroom
        ],
        "kitchen": [
            "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136",  # Modern kitchen
            "https://images.unsplash.com/photo-1556909172-54557c7e4fb7",  # Clean kitchen
            "https://images.unsplash.com/photo-1600585154340-be6161a56a0c",  # Organized kitchen
        ],
        "stairs": [
            "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c",  # Modern stairs
            "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea",  # Well-lit stairs
        ],
        "living_room": [
            "https://images.unsplash.com/photo-1616486338812-3dadae4b4ace",  # Clean living
            "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0",  # Organized space
        ]
    },
    "high_risk": {
        "bathroom": [
            "https://images.unsplash.com/photo-1584622781564-1d987f7333c1",  # Old bathroom
        ],
        "kitchen": [
            "https://images.unsplash.com/photo-1556909190-eccf4a8bf97a",  # Cluttered kitchen
        ],
        "stairs": [
            "https://images.unsplash.com/photo-1600585153490-76fb20a32601",  # Old stairs
        ],
        "living_room": [
            "https://images.unsplash.com/photo-1558618666-fcd25c85cd64",  # Cluttered room
        ]
    }
}


class ImageDownloader:
    """Downloads images from Unsplash API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = "data/curated"
    ):
        self.api_key = api_key or os.getenv("UNSPLASH_ACCESS_KEY")
        self.output_dir = Path(output_dir)
        self.base_url = "https://api.unsplash.com"
        self.metadata = {"images": [], "downloaded_at": None}

    def search_images(
        self,
        query: str,
        per_page: int = 10
    ) -> List[Dict]:
        """Search for images on Unsplash."""
        if not self.api_key:
            print("Warning: No API key. Using curated URLs instead.")
            return []

        try:
            response = requests.get(
                f"{self.base_url}/search/photos",
                params={
                    "query": query,
                    "per_page": per_page,
                    "orientation": "landscape"
                },
                headers={
                    "Authorization": f"Client-ID {self.api_key}"
                },
                timeout=30
            )

            if response.status_code == 403:
                print("API rate limit reached. Waiting 60 seconds...")
                time.sleep(60)
                return self.search_images(query, per_page)

            response.raise_for_status()
            data = response.json()
            return data.get("results", [])

        except requests.exceptions.RequestException as e:
            print(f"Error searching: {e}")
            return []

    def download_image(
        self,
        url: str,
        save_path: Path,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Download a single image."""
        try:
            # Get the actual image URL (regular size)
            if "unsplash.com/photos" in url or "images.unsplash.com" in url:
                # Direct URL - add size parameter
                if "?" in url:
                    image_url = f"{url}&w=1200&q=80"
                else:
                    image_url = f"{url}?w=1200&q=80"
            else:
                image_url = url

            response = requests.get(image_url, timeout=60)
            response.raise_for_status()

            # Save image
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(response.content)

            # Record metadata
            if metadata:
                self.metadata["images"].append({
                    "filename": save_path.name,
                    "path": str(save_path),
                    "source_url": url,
                    "photographer": metadata.get("user", {}).get("name", "Unknown"),
                    "unsplash_link": metadata.get("links", {}).get("html", url),
                    "license": "Unsplash License"
                })

            return True

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def download_category(
        self,
        risk_level: str,
        room_type: str,
        images_per_query: int = 3
    ) -> int:
        """Download images for a specific category."""
        queries = SEARCH_QUERIES.get(risk_level, {}).get(room_type, [])
        output_path = self.output_dir / risk_level / room_type

        downloaded = 0
        image_num = 1

        print(f"\nDownloading {risk_level}/{room_type}...")

        if self.api_key:
            # Use API
            for query in queries:
                print(f"  Searching: {query}")
                results = self.search_images(query, per_page=images_per_query)

                for result in results:
                    url = result.get("urls", {}).get("regular")
                    if not url:
                        continue

                    filename = f"{risk_level}_{room_type}_{image_num:03d}.jpg"
                    save_path = output_path / filename

                    if self.download_image(url, save_path, result):
                        print(f"    Downloaded: {filename}")
                        downloaded += 1
                        image_num += 1

                    # Rate limiting
                    time.sleep(0.5)

        else:
            # Use curated URLs (no API)
            urls = CURATED_URLS.get(risk_level, {}).get(room_type, [])
            for url in urls:
                filename = f"{risk_level}_{room_type}_{image_num:03d}.jpg"
                save_path = output_path / filename

                if self.download_image(url, save_path):
                    print(f"    Downloaded: {filename}")
                    downloaded += 1
                    image_num += 1

        return downloaded

    def download_all(self, images_per_query: int = 3) -> Dict[str, int]:
        """Download all categories."""
        stats = {}

        for risk_level in ["low_risk", "high_risk"]:
            for room_type in ["bathroom", "kitchen", "stairs", "living_room"]:
                key = f"{risk_level}/{room_type}"
                count = self.download_category(risk_level, room_type, images_per_query)
                stats[key] = count

        # Save metadata
        self.metadata["downloaded_at"] = datetime.now().isoformat()
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return stats

    def list_urls(self):
        """List all curated URLs without downloading."""
        print("\n" + "=" * 60)
        print("CURATED IMAGE URLs (No API Key Required)")
        print("=" * 60)

        for risk_level, rooms in CURATED_URLS.items():
            print(f"\n{risk_level.upper()}:")
            for room_type, urls in rooms.items():
                print(f"  {room_type}:")
                for url in urls:
                    print(f"    - {url}")

        print("\n" + "=" * 60)
        print("To download manually:")
        print("1. Visit each URL in your browser")
        print("2. Click 'Download free' button")
        print(f"3. Save to: data/curated/[risk_level]/[room_type]/")
        print("=" * 60)


def create_annotation_template(output_dir: Path):
    """Create annotation template for curated dataset."""
    annotations = {
        "dataset": "curated_risk_levels",
        "description": "Curated images for fall hazard detection research",
        "risk_levels": ["low_risk", "high_risk"],
        "room_types": ["bathroom", "kitchen", "stairs", "living_room"],
        "created_at": datetime.now().isoformat(),
        "annotations": {}
    }

    # Scan for downloaded images
    for risk_level in ["low_risk", "high_risk"]:
        risk_dir = output_dir / risk_level
        if not risk_dir.exists():
            continue

        for room_type in ["bathroom", "kitchen", "stairs", "living_room"]:
            room_dir = risk_dir / room_type
            if not room_dir.exists():
                continue

            for img_file in room_dir.glob("*.jpg"):
                image_id = img_file.stem

                # Create annotation entry
                if risk_level == "low_risk":
                    # Low risk = few or no hazards
                    annotations["annotations"][image_id] = {
                        "risk_level": "low",
                        "room_type": room_type,
                        "hazards": [],
                        "notes": "Low risk environment - minimal hazards expected"
                    }
                else:
                    # High risk = expected hazards (to be filled manually)
                    annotations["annotations"][image_id] = {
                        "risk_level": "high",
                        "room_type": room_type,
                        "hazards": [
                            {
                                "category": room_type if room_type in ["bathroom", "kitchen", "stairs"] else "general",
                                "subcategory": "to_be_annotated",
                                "severity": "medium",
                                "description": "Requires manual annotation"
                            }
                        ],
                        "notes": "High risk environment - needs manual hazard annotation"
                    }

    # Save annotations
    ann_dir = output_dir.parent / "annotations" / "curated"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_file = ann_dir / "annotations.json"

    with open(ann_file, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nAnnotation template saved to: {ann_file}")
    return ann_file


def main():
    parser = argparse.ArgumentParser(
        description='Download curated images for fall hazard detection research'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv("UNSPLASH_ACCESS_KEY"),
        help='Unsplash API access key'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/curated',
        help='Output directory for images'
    )
    parser.add_argument(
        '--risk-level',
        type=str,
        choices=['low_risk', 'high_risk'],
        help='Download only this risk level'
    )
    parser.add_argument(
        '--room',
        type=str,
        choices=['bathroom', 'kitchen', 'stairs', 'living_room'],
        help='Download only this room type'
    )
    parser.add_argument(
        '--images-per-query',
        type=int,
        default=3,
        help='Images to download per search query'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='List curated URLs without downloading'
    )
    parser.add_argument(
        '--create-annotations',
        action='store_true',
        help='Create annotation template after download'
    )

    args = parser.parse_args()

    downloader = ImageDownloader(
        api_key=args.api_key,
        output_dir=args.output_dir
    )

    if args.list_only:
        downloader.list_urls()
        return

    print("=" * 60)
    print("CURATED IMAGE DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"API key: {'Set' if args.api_key else 'Not set (using curated URLs)'}")

    if args.risk_level and args.room:
        # Download specific category
        count = downloader.download_category(
            args.risk_level,
            args.room,
            args.images_per_query
        )
        print(f"\nDownloaded {count} images")
    else:
        # Download all
        stats = downloader.download_all(args.images_per_query)

        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        total = 0
        for category, count in stats.items():
            print(f"  {category}: {count} images")
            total += count
        print(f"\nTotal: {total} images")

    if args.create_annotations:
        create_annotation_template(Path(args.output_dir))

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Review downloaded images")
    print("2. Run annotation script or manually annotate hazards")
    print("3. Run evaluation:")
    print("   python -m src.evaluation.run_evaluation \\")
    print("       --model llava moondream \\")
    print("       --data-dir data/curated/high_risk")
    print("=" * 60)


if __name__ == '__main__':
    main()
