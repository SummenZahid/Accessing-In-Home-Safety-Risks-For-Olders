"""
Westmead Home Safety Assessment - Hazard Categories

Comprehensive hazard definitions aligned with the Westmead Home Safety
Assessment Short Form for fall risk detection in elderly home environments.

Reference: Clemson, L. (1997, 2015). Westmead Home Safety Assessment.

Categories cover:
- External trafficways (gates, pathways, steps, ramps)
- General indoor areas (lighting, flooring, obstacles)
- Stairs and elevators
- Living areas (furniture, seating)
- Bedroom
- Bathroom
- Toilet area
- Kitchen
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional


class HazardCategory(Enum):
    """
    High-level hazard categories aligned with Westmead Assessment sections.

    Each category represents a major area of fall risk assessment.
    """
    FLOORING = "flooring"
    LIGHTING = "lighting"
    OBSTACLES = "obstacles"
    FURNITURE = "furniture"
    STAIRS = "stairs"
    BATHROOM = "bathroom"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    EXTERNAL = "external"
    GENERAL = "general"


@dataclass
class HazardSubcategory:
    """
    Detailed hazard subcategory with clinical metadata.

    Attributes:
        id: Unique identifier for the hazard type
        name: Human-readable hazard name
        category: Parent category
        description: Clinical description of the hazard
        clinical_severity_weight: Base severity weight (0-1) from clinical evidence
        westmead_section: Original Westmead assessment section reference
        examples: Example manifestations of this hazard
        detection_keywords: Keywords to help identify this hazard in descriptions
    """
    id: str
    name: str
    category: HazardCategory
    description: str
    clinical_severity_weight: float
    westmead_section: str
    examples: List[str]
    detection_keywords: List[str]


# =============================================================================
# COMPREHENSIVE HAZARD DEFINITIONS
# Based on Westmead Home Safety Assessment Short Form
# =============================================================================

HAZARD_DEFINITIONS: Dict[str, HazardSubcategory] = {

    # =========================================================================
    # FLOORING HAZARDS
    # =========================================================================

    "floor_loose_rug": HazardSubcategory(
        id="floor_loose_rug",
        name="Loose/Unsecured Rugs",
        category=HazardCategory.FLOORING,
        description="Rugs, mats, or carpet runners without non-slip backing that can slip, curl at edges, or bunch up underfoot",
        clinical_severity_weight=0.85,
        westmead_section="General Indoors - Floor Mats",
        examples=[
            "throw rugs without grip tape",
            "bath mats without suction cups",
            "hallway runners that slide",
            "rugs with curled or frayed edges",
            "small lightweight mats"
        ],
        detection_keywords=["rug", "mat", "carpet", "runner", "loose", "curled", "edge"]
    ),

    "floor_slippery": HazardSubcategory(
        id="floor_slippery",
        name="Slippery Floor Surface",
        category=HazardCategory.FLOORING,
        description="Floor surfaces that are smooth, polished, wet, or waxed creating slip hazards",
        clinical_severity_weight=0.90,
        westmead_section="General Indoors - Floors & Floor Coverings",
        examples=[
            "polished tile or marble floors",
            "freshly waxed hardwood",
            "wet floors near sinks or tubs",
            "glossy vinyl or linoleum",
            "smooth concrete"
        ],
        detection_keywords=["slippery", "wet", "polished", "shiny", "glossy", "smooth", "waxed"]
    ),

    "floor_uneven": HazardSubcategory(
        id="floor_uneven",
        name="Uneven Floor Surface",
        category=HazardCategory.FLOORING,
        description="Changes in floor level, damaged flooring, raised thresholds, or transitions between surfaces",
        clinical_severity_weight=0.80,
        westmead_section="General Indoors - Floors & Floor Coverings",
        examples=[
            "raised door thresholds",
            "cracked or broken tiles",
            "warped floorboards",
            "sunken areas in flooring",
            "transitions between carpet and tile"
        ],
        detection_keywords=["uneven", "raised", "threshold", "crack", "transition", "level change"]
    ),

    "floor_worn": HazardSubcategory(
        id="floor_worn",
        name="Worn Floor Covering",
        category=HazardCategory.FLOORING,
        description="Carpet, flooring, or mats that are worn, torn, or have loose edges that can catch feet",
        clinical_severity_weight=0.75,
        westmead_section="General Indoors - Floors & Floor Coverings",
        examples=[
            "frayed carpet edges",
            "torn linoleum lifting at edges",
            "worn carpet with holes",
            "peeling floor tiles"
        ],
        detection_keywords=["worn", "torn", "frayed", "peeling", "lifting", "damaged"]
    ),

    # =========================================================================
    # LIGHTING HAZARDS
    # =========================================================================

    "light_insufficient": HazardSubcategory(
        id="light_insufficient",
        name="Insufficient Lighting",
        category=HazardCategory.LIGHTING,
        description="Areas with inadequate illumination making it difficult to see floor hazards or judge distances",
        clinical_severity_weight=0.75,
        westmead_section="General Indoors - Lighting",
        examples=[
            "dark hallways",
            "dimly lit stairways",
            "unlit entrances",
            "shadow areas in rooms",
            "insufficient natural light"
        ],
        detection_keywords=["dark", "dim", "shadow", "unlit", "poor lighting", "insufficient light"]
    ),

    "light_night_absent": HazardSubcategory(
        id="light_night_absent",
        name="Missing Night Lighting",
        category=HazardCategory.LIGHTING,
        description="Lack of night lights or illumination for nighttime navigation, especially path to bathroom",
        clinical_severity_weight=0.80,
        westmead_section="External Trafficways - Night Lighting",
        examples=[
            "no night light in hallway",
            "dark path to bathroom",
            "unlit bedroom floor",
            "no sensor lights"
        ],
        detection_keywords=["night light", "sensor", "nighttime", "after dark"]
    ),

    "light_glare": HazardSubcategory(
        id="light_glare",
        name="Excessive Glare",
        category=HazardCategory.LIGHTING,
        description="Bright light sources or reflections causing visual discomfort or temporary blindness",
        clinical_severity_weight=0.60,
        westmead_section="General Indoors - Lighting",
        examples=[
            "uncovered windows with direct sunlight",
            "highly reflective floor surfaces",
            "exposed light bulbs",
            "reflections from mirrors"
        ],
        detection_keywords=["glare", "bright", "reflection", "sunlight", "exposed bulb"]
    ),

    "light_switch_access": HazardSubcategory(
        id="light_switch_access",
        name="Inaccessible Light Switches",
        category=HazardCategory.LIGHTING,
        description="Light switches not positioned at doorways or not reachable without walking in the dark",
        clinical_severity_weight=0.65,
        westmead_section="General Indoors - Lighting",
        examples=[
            "switch far from door entrance",
            "no switch at top or bottom of stairs",
            "bedside lamp without easy switch"
        ],
        detection_keywords=["switch", "access", "reach", "doorway"]
    ),

    # =========================================================================
    # OBSTACLE HAZARDS
    # =========================================================================

    "obstacle_cord": HazardSubcategory(
        id="obstacle_cord",
        name="Trailing Cords/Cables",
        category=HazardCategory.OBSTACLES,
        description="Electrical cords, cables, or wires running across walkways creating trip hazards",
        clinical_severity_weight=0.90,
        westmead_section="General Indoors - Space",
        examples=[
            "extension cords across floor",
            "phone charger cables in walkway",
            "lamp cords crossing path",
            "TV cables on floor"
        ],
        detection_keywords=["cord", "cable", "wire", "extension", "trailing", "across"]
    ),

    "obstacle_clutter": HazardSubcategory(
        id="obstacle_clutter",
        name="Floor Clutter",
        category=HazardCategory.OBSTACLES,
        description="Items, objects, or belongings left on the floor creating trip hazards",
        clinical_severity_weight=0.85,
        westmead_section="General Indoors - Tidiness/Cleanliness",
        examples=[
            "newspapers or magazines on floor",
            "shoes left in walkways",
            "bags or boxes on floor",
            "toys or pet items",
            "stacked items near walking paths"
        ],
        detection_keywords=["clutter", "items", "objects", "mess", "pile", "stack"]
    ),

    "obstacle_pathway": HazardSubcategory(
        id="obstacle_pathway",
        name="Obstructed Pathway",
        category=HazardCategory.OBSTACLES,
        description="Furniture, objects, or items blocking normal walking routes through rooms",
        clinical_severity_weight=0.85,
        westmead_section="General Indoors - Space",
        examples=[
            "furniture blocking doorway",
            "items in hallway",
            "narrow passages between furniture",
            "boxes blocking routes"
        ],
        detection_keywords=["blocked", "narrow", "obstruct", "crowded", "tight"]
    ),

    "obstacle_spill": HazardSubcategory(
        id="obstacle_spill",
        name="Spills on Floor",
        category=HazardCategory.OBSTACLES,
        description="Liquid spills or wet spots on the floor creating slip hazards",
        clinical_severity_weight=0.85,
        westmead_section="General Indoors - Tidiness/Cleanliness",
        examples=[
            "water spill near sink",
            "pet water bowl overflow",
            "wet spots from plants",
            "tracked-in water near entrance"
        ],
        detection_keywords=["spill", "wet", "liquid", "water", "puddle"]
    ),

    # =========================================================================
    # FURNITURE HAZARDS
    # =========================================================================

    "furniture_unstable": HazardSubcategory(
        id="furniture_unstable",
        name="Unstable Furniture",
        category=HazardCategory.FURNITURE,
        description="Furniture that could tip, wobble, or collapse when used for support while walking",
        clinical_severity_weight=0.85,
        westmead_section="Living Area - Furniture",
        examples=[
            "wobbly side tables",
            "lightweight chairs",
            "furniture on wheels without locks",
            "glass tables",
            "rickety shelving"
        ],
        detection_keywords=["unstable", "wobbly", "wheels", "lightweight", "glass"]
    ),

    "furniture_sharp_edges": HazardSubcategory(
        id="furniture_sharp_edges",
        name="Sharp Furniture Edges",
        category=HazardCategory.FURNITURE,
        description="Furniture with sharp corners or edges at body height that could cause injury in a fall",
        clinical_severity_weight=0.60,
        westmead_section="Living Area - Furniture",
        examples=[
            "glass coffee table corners",
            "sharp corner of dresser",
            "metal table edges",
            "protruding cabinet hardware"
        ],
        detection_keywords=["sharp", "corner", "edge", "glass", "protruding"]
    ),

    "furniture_seat_height": HazardSubcategory(
        id="furniture_seat_height",
        name="Inappropriate Seat Height",
        category=HazardCategory.FURNITURE,
        description="Seating that is too low, too soft, or lacks proper support for safe sitting and standing",
        clinical_severity_weight=0.70,
        westmead_section="Living Area - Seating",
        examples=[
            "low sofa difficult to rise from",
            "deep soft cushions",
            "chair without armrests",
            "high bar stools"
        ],
        detection_keywords=["low", "soft", "deep", "height", "armrest"]
    ),

    "furniture_support_used": HazardSubcategory(
        id="furniture_support_used",
        name="Unstable Support Objects",
        category=HazardCategory.FURNITURE,
        description="Objects that might be used for support but are unstable or not designed for it",
        clinical_severity_weight=0.80,
        westmead_section="General Indoors - Space",
        examples=[
            "towel bars used as grab bars",
            "stepping on furniture to reach",
            "using curtains for support",
            "leaning on unstable shelves"
        ],
        detection_keywords=["support", "grab", "lean", "reach", "stepping"]
    ),

    # =========================================================================
    # STAIRS HAZARDS
    # =========================================================================

    "stairs_no_handrail": HazardSubcategory(
        id="stairs_no_handrail",
        name="Missing Handrails",
        category=HazardCategory.STAIRS,
        description="Stairs or steps without adequate handrails on one or both sides",
        clinical_severity_weight=0.95,
        westmead_section="Stairs/Elevators - Handrails",
        examples=[
            "stairs with no banister",
            "handrail only on one side",
            "steps without any rail",
            "porch steps without support"
        ],
        detection_keywords=["handrail", "banister", "rail", "missing", "none"]
    ),

    "stairs_handrail_poor": HazardSubcategory(
        id="stairs_handrail_poor",
        name="Inadequate Handrails",
        category=HazardCategory.STAIRS,
        description="Handrails that are loose, poorly positioned, too short, or in poor condition",
        clinical_severity_weight=0.85,
        westmead_section="Stairs/Elevators - Handrails",
        examples=[
            "loose or wobbly handrail",
            "handrail that doesn't extend full length",
            "rail too far from stairs",
            "damaged or rough handrail"
        ],
        detection_keywords=["loose", "wobbly", "short", "position", "condition"]
    ),

    "stairs_slippery": HazardSubcategory(
        id="stairs_slippery",
        name="Slippery Stairs",
        category=HazardCategory.STAIRS,
        description="Stair treads that are smooth, polished, or have worn coverings creating slip risk",
        clinical_severity_weight=0.90,
        westmead_section="Stairs/Elevators - Steps/Stairs",
        examples=[
            "polished wood stairs without treads",
            "worn carpet on stairs",
            "smooth tile steps",
            "stairs with loose covering"
        ],
        detection_keywords=["slippery", "smooth", "worn", "polished", "tread"]
    ),

    "stairs_visibility": HazardSubcategory(
        id="stairs_visibility",
        name="Poor Stair Visibility",
        category=HazardCategory.STAIRS,
        description="Stairs difficult to see due to lighting, color, or lack of edge contrast",
        clinical_severity_weight=0.80,
        westmead_section="Stairs/Elevators - Steps/Stairs",
        examples=[
            "no contrast on stair edges",
            "dark stairwell",
            "patterned carpet obscuring edges",
            "stairs same color as landing"
        ],
        detection_keywords=["dark", "contrast", "edge", "visibility", "pattern"]
    ),

    "stairs_dimensions": HazardSubcategory(
        id="stairs_dimensions",
        name="Poor Stair Dimensions",
        category=HazardCategory.STAIRS,
        description="Stairs that are too high, have uneven heights, deep treads, or overhang",
        clinical_severity_weight=0.75,
        westmead_section="Stairs/Elevators - Steps/Stairs",
        examples=[
            "steps too high",
            "uneven step heights",
            "narrow treads",
            "step overhang catching toes"
        ],
        detection_keywords=["high", "uneven", "deep", "narrow", "overhang"]
    ),

    # =========================================================================
    # BATHROOM HAZARDS
    # =========================================================================

    "bath_no_grab_bars": HazardSubcategory(
        id="bath_no_grab_bars",
        name="Missing Grab Bars",
        category=HazardCategory.BATHROOM,
        description="No grab rails near toilet, shower, or bathtub for support",
        clinical_severity_weight=0.95,
        westmead_section="Bathroom - Grabrails",
        examples=[
            "no grab bar by toilet",
            "shower without grab bars",
            "bathtub without support rails",
            "no support for tub entry"
        ],
        detection_keywords=["grab bar", "rail", "support", "handle", "missing"]
    ),

    "bath_grab_bars_poor": HazardSubcategory(
        id="bath_grab_bars_poor",
        name="Inadequate Grab Bars",
        category=HazardCategory.BATHROOM,
        description="Grab bars that are poorly positioned, wrong angle, too short, or not secure",
        clinical_severity_weight=0.80,
        westmead_section="Bathroom - Grabrails",
        examples=[
            "loose grab bar",
            "grab bar at wrong height",
            "towel bar used instead of grab bar",
            "bar too far from toilet"
        ],
        detection_keywords=["loose", "position", "height", "angle", "secure"]
    ),

    "bath_floor_slippery": HazardSubcategory(
        id="bath_floor_slippery",
        name="Slippery Bathroom Floor",
        category=HazardCategory.BATHROOM,
        description="Bathroom floor surfaces that are slippery when wet or dry",
        clinical_severity_weight=0.90,
        westmead_section="Bathroom - Floor Surface",
        examples=[
            "smooth tile floor",
            "wet bathroom floor",
            "loose bath mat",
            "glossy floor surface"
        ],
        detection_keywords=["slippery", "wet", "smooth", "tile", "mat"]
    ),

    "bath_tub_entry": HazardSubcategory(
        id="bath_tub_entry",
        name="Difficult Tub/Shower Entry",
        category=HazardCategory.BATHROOM,
        description="High step or barrier required to enter shower or bathtub",
        clinical_severity_weight=0.85,
        westmead_section="Bathroom - Shower Recess / Bath",
        examples=[
            "high bathtub wall",
            "shower with high threshold",
            "step-over into tub",
            "narrow shower entry"
        ],
        detection_keywords=["high", "step", "threshold", "entry", "barrier"]
    ),

    "bath_toilet_height": HazardSubcategory(
        id="bath_toilet_height",
        name="Inappropriate Toilet Height",
        category=HazardCategory.BATHROOM,
        description="Toilet that is too low or too high for safe transfers",
        clinical_severity_weight=0.70,
        westmead_section="Toilet Area - Toilet",
        examples=[
            "low toilet difficult to rise from",
            "standard height toilet without raised seat",
            "toilet too far from grab bar"
        ],
        detection_keywords=["toilet", "height", "low", "rise", "transfer"]
    ),

    "bath_floor_covering": HazardSubcategory(
        id="bath_floor_covering",
        name="Poor Bathroom Floor Covering",
        category=HazardCategory.BATHROOM,
        description="Slippery mats, worn floor covering, or raised/loose tiles in bathroom",
        clinical_severity_weight=0.80,
        westmead_section="Bathroom - Floor Surface",
        examples=[
            "bath mat without suction",
            "raised tile edges",
            "worn vinyl flooring",
            "curled mat edges"
        ],
        detection_keywords=["mat", "tile", "worn", "loose", "curled"]
    ),

    # =========================================================================
    # BEDROOM HAZARDS
    # =========================================================================

    "bedroom_bed_height": HazardSubcategory(
        id="bedroom_bed_height",
        name="Inappropriate Bed Height",
        category=HazardCategory.BEDROOM,
        description="Bed that is too high or too low for safe transfers",
        clinical_severity_weight=0.70,
        westmead_section="Bedroom - Bed",
        examples=[
            "low platform bed",
            "bed too high requiring step",
            "soft mattress making rising difficult",
            "bed without stable edge"
        ],
        detection_keywords=["bed", "height", "low", "high", "transfer"]
    ),

    "bedroom_path_bathroom": HazardSubcategory(
        id="bedroom_path_bathroom",
        name="Hazardous Path to Bathroom",
        category=HazardCategory.BEDROOM,
        description="Obstacles or hazards in nighttime path from bed to bathroom",
        clinical_severity_weight=0.85,
        westmead_section="Bedroom / Toilet Area - Location",
        examples=[
            "furniture blocking path",
            "no clear path to bathroom",
            "obstacles in walking route",
            "dark path to bathroom"
        ],
        detection_keywords=["path", "bathroom", "route", "night", "obstacle"]
    ),

    "bedroom_lighting": HazardSubcategory(
        id="bedroom_lighting",
        name="Poor Bedroom Lighting",
        category=HazardCategory.BEDROOM,
        description="Inadequate bedside lighting or no accessible light switch",
        clinical_severity_weight=0.70,
        westmead_section="Bedroom - Bed Lighting",
        examples=[
            "no bedside lamp",
            "lamp switch hard to reach",
            "no night light",
            "light switch far from bed"
        ],
        detection_keywords=["bedside", "lamp", "light", "switch", "night"]
    ),

    "bedroom_trailing": HazardSubcategory(
        id="bedroom_trailing",
        name="Trailing Items in Bedroom",
        category=HazardCategory.BEDROOM,
        description="Curtains, bedcovers, or items trailing in walking paths",
        clinical_severity_weight=0.65,
        westmead_section="Bedroom - Curtains/Bed Covers",
        examples=[
            "long curtains on floor",
            "bedspread trailing on floor",
            "throw blankets in walkway",
            "electrical cords from nightstand"
        ],
        detection_keywords=["trailing", "curtain", "bedspread", "blanket", "cord"]
    ),

    # =========================================================================
    # KITCHEN HAZARDS
    # =========================================================================

    "kitchen_high_storage": HazardSubcategory(
        id="kitchen_high_storage",
        name="High Storage Access",
        category=HazardCategory.KITCHEN,
        description="Commonly used items stored in high cabinets requiring reaching or climbing",
        clinical_severity_weight=0.70,
        westmead_section="Kitchen - Kitchen Equipment",
        examples=[
            "dishes in upper cabinets",
            "items stored above reach",
            "need to use step stool",
            "heavy items stored high"
        ],
        detection_keywords=["high", "reach", "cabinet", "storage", "upper"]
    ),

    "kitchen_spill_areas": HazardSubcategory(
        id="kitchen_spill_areas",
        name="Spill-Prone Areas",
        category=HazardCategory.KITCHEN,
        description="Areas near sink, stove, or refrigerator prone to water or grease spills",
        clinical_severity_weight=0.75,
        westmead_section="Kitchen - Kitchen Equipment",
        examples=[
            "wet floor near sink",
            "grease near stove",
            "water near refrigerator",
            "spills not cleaned up"
        ],
        detection_keywords=["sink", "stove", "spill", "wet", "grease"]
    ),

    "kitchen_floor": HazardSubcategory(
        id="kitchen_floor",
        name="Kitchen Floor Hazards",
        category=HazardCategory.KITCHEN,
        description="Slippery kitchen floors, mats without grip, or obstacles in cooking area",
        clinical_severity_weight=0.80,
        westmead_section="Kitchen - Kitchen Equipment",
        examples=[
            "slippery tile floor",
            "mat near sink without grip",
            "items stored on floor",
            "wet spots from cooking"
        ],
        detection_keywords=["floor", "mat", "slippery", "wet", "tile"]
    ),

    # =========================================================================
    # EXTERNAL HAZARDS
    # =========================================================================

    "external_pathway": HazardSubcategory(
        id="external_pathway",
        name="External Pathway Hazards",
        category=HazardCategory.EXTERNAL,
        description="Uneven, slippery, or obstructed outdoor pathways and driveways",
        clinical_severity_weight=0.80,
        westmead_section="External Trafficways - Pathways/Driveways",
        examples=[
            "cracked pavement",
            "uneven paving stones",
            "slippery when wet",
            "overgrown vegetation",
            "steep gradient"
        ],
        detection_keywords=["path", "driveway", "crack", "uneven", "slippery"]
    ),

    "external_steps": HazardSubcategory(
        id="external_steps",
        name="External Step Hazards",
        category=HazardCategory.EXTERNAL,
        description="Outdoor steps without handrails, uneven, or slippery",
        clinical_severity_weight=0.85,
        westmead_section="External Trafficways - Steps",
        examples=[
            "porch steps without rail",
            "slippery outdoor steps",
            "uneven step heights",
            "steps with poor visibility"
        ],
        detection_keywords=["step", "porch", "outdoor", "rail", "slippery"]
    ),

    "external_lighting": HazardSubcategory(
        id="external_lighting",
        name="External Lighting",
        category=HazardCategory.EXTERNAL,
        description="Inadequate outdoor lighting for pathways, steps, or entrances",
        clinical_severity_weight=0.75,
        westmead_section="External Trafficways - Night Lighting",
        examples=[
            "dark entrance area",
            "unlit pathway",
            "no motion sensor lights",
            "shadowy areas at night"
        ],
        detection_keywords=["outdoor", "entrance", "light", "dark", "pathway"]
    ),

    "external_doormat": HazardSubcategory(
        id="external_doormat",
        name="Doormat Hazards",
        category=HazardCategory.EXTERNAL,
        description="Doormats with curled edges, worn areas, or slippery surfaces",
        clinical_severity_weight=0.70,
        westmead_section="External Trafficways - Doormat",
        examples=[
            "curled doormat edges",
            "worn doormat",
            "slippery mat",
            "mat without grip"
        ],
        detection_keywords=["doormat", "mat", "entrance", "curled", "worn"]
    ),

    # =========================================================================
    # GENERAL HAZARDS
    # =========================================================================

    "general_mobility_aid": HazardSubcategory(
        id="general_mobility_aid",
        name="Mobility Aid Issues",
        category=HazardCategory.GENERAL,
        description="Walking aids that are inappropriate, in poor condition, or not accessible",
        clinical_severity_weight=0.75,
        westmead_section="General Indoors - Mobility Aid",
        examples=[
            "walker with worn tips",
            "cane wrong height",
            "mobility aid stored far from bed",
            "aid in poor condition"
        ],
        detection_keywords=["walker", "cane", "mobility", "aid", "condition"]
    ),

    "general_reaching": HazardSubcategory(
        id="general_reaching",
        name="Unsafe Reaching",
        category=HazardCategory.GENERAL,
        description="Need to reach for items using unstable support or while standing on furniture",
        clinical_severity_weight=0.75,
        westmead_section="General Indoors - Reaching for High Places",
        examples=[
            "standing on chair to reach",
            "using unstable stool",
            "overreaching while standing",
            "climbing on furniture"
        ],
        detection_keywords=["reach", "climb", "stool", "step", "high"]
    ),

    "general_pets": HazardSubcategory(
        id="general_pets",
        name="Pet-Related Hazards",
        category=HazardCategory.GENERAL,
        description="Small or playful pets that may get underfoot or cause tripping",
        clinical_severity_weight=0.60,
        westmead_section="Living Area - Pets",
        examples=[
            "small dog underfoot",
            "cat that weaves between legs",
            "pet toys on floor",
            "pet food bowls in walkway"
        ],
        detection_keywords=["pet", "dog", "cat", "animal", "toy", "bowl"]
    ),

    "general_footwear": HazardSubcategory(
        id="general_footwear",
        name="Inappropriate Footwear",
        category=HazardCategory.GENERAL,
        description="Footwear that is worn, slippery, ill-fitting, or inappropriate for home use",
        clinical_severity_weight=0.70,
        westmead_section="Living Area - Footwear",
        examples=[
            "loose slippers",
            "worn shoe soles",
            "high heels indoors",
            "socks on slippery floor",
            "flip flops or slides"
        ],
        detection_keywords=["shoe", "slipper", "footwear", "heel", "sock", "barefoot"]
    ),
}


def get_hazards_by_category(category: HazardCategory) -> List[HazardSubcategory]:
    """
    Get all hazard subcategories for a specific category.

    Args:
        category: The HazardCategory to filter by

    Returns:
        List of HazardSubcategory objects for that category
    """
    return [h for h in HAZARD_DEFINITIONS.values() if h.category == category]


def get_room_priority_categories(room_type: str) -> List[HazardCategory]:
    """
    Get prioritized hazard categories based on room type.

    Args:
        room_type: Type of room (bathroom, kitchen, bedroom, etc.)

    Returns:
        List of HazardCategory in priority order for that room
    """
    room_priorities = {
        "bathroom": [
            HazardCategory.BATHROOM,
            HazardCategory.FLOORING,
            HazardCategory.LIGHTING,
            HazardCategory.OBSTACLES,
        ],
        "kitchen": [
            HazardCategory.KITCHEN,
            HazardCategory.FLOORING,
            HazardCategory.OBSTACLES,
            HazardCategory.LIGHTING,
        ],
        "bedroom": [
            HazardCategory.BEDROOM,
            HazardCategory.LIGHTING,
            HazardCategory.FLOORING,
            HazardCategory.OBSTACLES,
        ],
        "living_room": [
            HazardCategory.FURNITURE,
            HazardCategory.OBSTACLES,
            HazardCategory.FLOORING,
            HazardCategory.LIGHTING,
        ],
        "hallway": [
            HazardCategory.LIGHTING,
            HazardCategory.FLOORING,
            HazardCategory.OBSTACLES,
        ],
        "stairs": [
            HazardCategory.STAIRS,
            HazardCategory.LIGHTING,
        ],
        "external": [
            HazardCategory.EXTERNAL,
            HazardCategory.LIGHTING,
        ],
    }

    return room_priorities.get(
        room_type.lower(),
        [HazardCategory.FLOORING, HazardCategory.OBSTACLES, HazardCategory.LIGHTING]
    )


def get_all_detection_keywords() -> Dict[str, str]:
    """
    Get mapping of all detection keywords to hazard IDs.

    Returns:
        Dictionary mapping keywords to hazard subcategory IDs
    """
    keyword_map = {}
    for hazard_id, hazard in HAZARD_DEFINITIONS.items():
        for keyword in hazard.detection_keywords:
            keyword_map[keyword.lower()] = hazard_id
    return keyword_map


def get_category_weights() -> Dict[HazardCategory, float]:
    """
    Get average clinical severity weight per category.

    Returns:
        Dictionary mapping categories to their average weights
    """
    from collections import defaultdict

    category_weights = defaultdict(list)
    for hazard in HAZARD_DEFINITIONS.values():
        category_weights[hazard.category].append(hazard.clinical_severity_weight)

    return {
        cat: sum(weights) / len(weights)
        for cat, weights in category_weights.items()
    }
