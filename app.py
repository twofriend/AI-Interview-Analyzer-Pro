
import io
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Velix GeoAI Phase 1",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Velix GeoAI Phase 1: Satellite Index Dashboard")
st.caption("Upload a Sentinel-2 style GeoTIFF and calculate NDVI, NDWI, NDBI, land-cover masks, and area statistics.")


# -----------------------------
# Helper Functions
# -----------------------------
def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=np.abs(denominator) > 1e-6
    )


def normalize_for_rgb(arr):
    arr = arr.astype(np.float32)
    p2, p98 = np.nanpercentile(arr, (2, 98))
    if p98 - p2 == 0:
        return np.zeros_like(arr)
    arr = (arr - p2) / (p98 - p2)
    return np.clip(arr, 0, 1)


def plot_array(array, title, cmap=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if cmap is not None:
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    st.pyplot(fig)


def get_pixel_area_km2(dataset):
    """
    Calculates pixel area in km².
    Best if GeoTIFF is in projected CRS such as UTM.
    If CRS is geographic degrees, area is approximate.
    """
    transform = dataset.transform
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    if dataset.crs and dataset.crs.is_projected:
        return (pixel_width * pixel_height) / 1_000_000

    # approximate conversion for degrees near image center latitude
    bounds = dataset.bounds
    center_lat = (bounds.top + bounds.bottom) / 2
    meters_per_degree_lat = 111_320
    meters_per_degree_lon = 111_320 * math.cos(math.radians(center_lat))
    return (pixel_width * meters_per_degree_lon * pixel_height * meters_per_degree_lat) / 1_000_000


def classify_indices(ndvi, ndwi, ndbi):
    """
    Simple threshold-based masks for Phase 1.
    These thresholds are starting points; tune them for Karachi using visual inspection in QGIS.
    """
    water = ndwi > 0.20
    vegetation = (ndvi > 0.30) & (~water)
    built_up = (ndbi > 0.10) & (ndvi < 0.25) & (~water)
    bare_land = (~water) & (~vegetation) & (~built_up)

    classified = np.zeros(ndvi.shape, dtype=np.uint8)
    classified[water] = 1
    classified[vegetation] = 2
    classified[built_up] = 3
    classified[bare_land] = 4

    return classified, water, vegetation, built_up, bare_land


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Band Mapping")

st.sidebar.markdown(
    """
    Recommended Sentinel-2 band order for a single multiband GeoTIFF:

    **Band 1 = Blue B2**  
    **Band 2 = Green B3**  
    **Band 3 = Red B4**  
    **Band 4 = NIR B8**  
    **Band 5 = SWIR B11**  
    """
)

blue_band = st.sidebar.number_input("Blue band index (B2)", min_value=1, value=1)
green_band = st.sidebar.number_input("Green band index (B3)", min_value=1, value=2)
red_band = st.sidebar.number_input("Red band index (B4)", min_value=1, value=3)
nir_band = st.sidebar.number_input("NIR band index (B8)", min_value=1, value=4)
swir_band = st.sidebar.number_input("SWIR band index (B11)", min_value=1, value=5)

st.sidebar.header("Thresholds")
ndvi_threshold = st.sidebar.slider("Vegetation NDVI threshold", -1.0, 1.0, 0.30, 0.01)
ndwi_threshold = st.sidebar.slider("Water NDWI threshold", -1.0, 1.0, 0.20, 0.01)
ndbi_threshold = st.sidebar.slider("Built-up NDBI threshold", -1.0, 1.0, 0.10, 0.01)


uploaded_file = st.file_uploader(
    "Upload Sentinel-2 multiband GeoTIFF (.tif / .tiff)",
    type=["tif", "tiff"]
)

if uploaded_file is None:
    st.info("Upload a multiband GeoTIFF to begin. For Phase 1, use Sentinel-2 bands B2, B3, B4, B8, and B11.")
    st.stop()


# -----------------------------
# Read GeoTIFF
# -----------------------------
with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

try:
    with rasterio.open(tmp_path) as src:
        band_count = src.count
        st.success(f"GeoTIFF loaded successfully. Bands found: {band_count}")

        needed_bands = [blue_band, green_band, red_band, nir_band, swir_band]
        if max(needed_bands) > band_count:
            st.error(
                f"Your file has only {band_count} bands, but your selected band mapping requires band {max(needed_bands)}."
            )
            st.stop()

        blue = src.read(int(blue_band)).astype(np.float32)
        green = src.read(int(green_band)).astype(np.float32)
        red = src.read(int(red_band)).astype(np.float32)
        nir = src.read(int(nir_band)).astype(np.float32)
        swir = src.read(int(swir_band)).astype(np.float32)

        profile = src.profile
        pixel_area_km2 = get_pixel_area_km2(src)

        bounds = src.bounds
        crs = src.crs

except Exception as e:
    st.error(f"Could not read GeoTIFF: {e}")
    st.stop()


# -----------------------------
# Calculate Indices
# -----------------------------
ndvi = safe_divide(nir - red, nir + red)
ndwi = safe_divide(green - nir, green + nir)
ndbi = safe_divide(swir - nir, swir + nir)

water = ndwi > ndwi_threshold
vegetation = (ndvi > ndvi_threshold) & (~water)
built_up = (ndbi > ndbi_threshold) & (ndvi < 0.25) & (~water)
bare_land = (~water) & (~vegetation) & (~built_up)

classified = np.zeros(ndvi.shape, dtype=np.uint8)
classified[water] = 1
classified[vegetation] = 2
classified[built_up] = 3
classified[bare_land] = 4

rgb = np.dstack([
    normalize_for_rgb(red),
    normalize_for_rgb(green),
    normalize_for_rgb(blue)
])


# -----------------------------
# Metadata
# -----------------------------
with st.expander("GeoTIFF Metadata"):
    st.write({
        "CRS": str(crs),
        "Bounds": {
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top,
        },
        "Width": profile["width"],
        "Height": profile["height"],
        "Pixel area km²": pixel_area_km2,
    })


# -----------------------------
# Visual Outputs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["RGB Image", "Index Maps", "Classified Map", "Area Report"])

with tab1:
    st.subheader("Natural Color RGB")
    plot_array(rgb, "RGB Composite")

with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        plot_array(ndvi, "NDVI: Vegetation Index", cmap="RdYlGn", vmin=-1, vmax=1)
    with c2:
        plot_array(ndwi, "NDWI: Water Index", cmap="Blues", vmin=-1, vmax=1)
    with c3:
        plot_array(ndbi, "NDBI: Built-up Index", cmap="inferno", vmin=-1, vmax=1)

with tab3:
    st.subheader("Simple Land Cover Classification")
    st.markdown(
        """
        Class codes:  
        **1 = Water**, **2 = Vegetation**, **3 = Built-up**, **4 = Bare land / Other**
        """
    )
    plot_array(classified, "Threshold-Based Land Cover Map", cmap="tab10", vmin=0, vmax=4)

with tab4:
    st.subheader("Area Statistics")

    total_pixels = classified.size
    total_area = total_pixels * pixel_area_km2

    stats = pd.DataFrame([
        ["Water", int(np.sum(water)), np.sum(water) * pixel_area_km2],
        ["Vegetation", int(np.sum(vegetation)), np.sum(vegetation) * pixel_area_km2],
        ["Built-up", int(np.sum(built_up)), np.sum(built_up) * pixel_area_km2],
        ["Bare land / Other", int(np.sum(bare_land)), np.sum(bare_land) * pixel_area_km2],
    ], columns=["Class", "Pixel Count", "Area km²"])

    stats["Percentage"] = (stats["Area km²"] / total_area) * 100
    stats["Area km²"] = stats["Area km²"].round(4)
    stats["Percentage"] = stats["Percentage"].round(2)

    st.dataframe(stats, use_container_width=True)

    csv = stats.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Area Statistics CSV",
        data=csv,
        file_name="geoai_phase1_area_statistics.csv",
        mime="text/csv"
    )

    report_text = f"""
Velix GeoAI Phase 1 Report

Total estimated area: {total_area:.4f} km²

Class-wise summary:
{stats.to_string(index=False)}

Interpretation:
- High NDVI areas indicate vegetation.
- High NDWI areas indicate water or wet surfaces.
- High NDBI areas indicate built-up or impervious urban surfaces.
- Thresholds should be tuned using QGIS visual inspection and local knowledge.
"""

    st.download_button(
        "Download Simple Text Report",
        data=report_text,
        file_name="geoai_phase1_report.txt",
        mime="text/plain"
    )


st.caption("Phase 1 prototype. For research-grade results, validate thresholds with ground truth or ESA WorldCover/Dynamic World labels.")
