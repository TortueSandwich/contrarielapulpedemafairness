from typing import Any, Dict, Union, Iterable
from numbers import Number
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image, ExifTags  # type: ignore
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def crop_image(img: Image.Image, crop_size: int = 32) -> Image.Image:
    """Crop a PIL image to crop_size x crop_size."""
    # First determine the bounding box
    width, height = img.size
    new_width = 1.0 * crop_size
    new_height = 1.0 * crop_size
    left = round((width - new_width) / 2)
    top = round((height - new_height) / 2)
    right = round((width + new_width) / 2)
    bottom = round((height + new_height) / 2)
    # Then crop the image to that box
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def image_to_vector(img: Image.Image) -> np.ndarray:
    """Return the values of all the pixels of the Image as a vector"""
    return np.array(img).ravel()


def image_to_series(img: Image.Image) -> np.ndarray:
    """Return the values of all the pixels of the Image as a series"""
    # Note: the transparency layer is ignored.
    return pd.Series(np.array(img)[:, :, :3].ravel())



def difference_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract a numpy array D = R-(G+B)/2 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    D = R - (G + B) / 2
    return D


def value_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract a numpy array V = (R+G+B)/3 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    V = (R + G + B) / 3
    return V


def foreground_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    M = np.array(img)  # In case this is not yet a Numpy array
    G = np.min(M[:, :, 0:3], axis=2)
    F = G < theta
    return F


def closest_non_transparent(M,position)->tuple:
    m=max(M.shape[0],M.shape[1])
    closest=None
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j,3]==255 and np.sqrt((i-position[0])**2+(j-position[1])**2)<m:
                m=np.sqrt((i-position[0])**2+(j-position[1])**2)
                closest=(i,j)
    return closest

def grayscale(img: Image.Image) -> np.ndarray:
    """Return image in gray scale"""
    return np.mean(np.array(img)[:, :, :3], axis=2)

class MatchedFilter:
    """Matched filter class to extract a feature from a template"""

    def __init__(self, examples: Iterable[Image.Image]):
        """Create the template for a matched filter; use only grayscale"""
        # Compute the average of all images after conversion to grayscale
        M = np.mean([grayscale(img) for img in examples], axis=0)
        # Standardize
        self.template = (M - M.mean()) / M.std()

    def show(self):
        """Show the template"""
        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot()
        ax.imshow(self.template, cmap="gray")
        return fig

    def match(self, img: Image.Image) -> Number:
        """Extract the matched filter value for a PIL image."""
        # Convert to grayscale and standardize
        M = grayscale(img)
        M = (M - M.mean()) / M.std()
        # Compute scalar product with the template
        # This reinforce black and white if they agree
        return np.mean(np.multiply(self.template, M))


def invert_if_light_background(img: np.ndarray) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    if np.count_nonzero(img) > np.count_nonzero(np.invert(img)):
        return np.invert(img)
    else:
        return img


def crop_around_center(img, center, crop_size=32, verbose=False):
    """Crop a PIL image to crop_size x crop_size."""
    # First determine the center of gravity
    x0, y0 = center[1], center[0]
    # Initialize the image with the old_style cropped image
    # cropped_img = crop_image(img)
    if verbose:
        print(x0, y0)
    # Determine the bounding box
    width, height = img.size
    if verbose:
        print(width, height)
    new_width = 1.0 * crop_size
    new_height = 1.0 * crop_size
    if verbose:
        print(new_width, new_height)
    left = round(x0 - new_width / 2)
    top = round(y0 - new_height / 2)
    right = round(x0 + new_width / 2)
    bottom = round(y0 + new_height / 2)
    if verbose:
        print(left, top, right, bottom)
    # Then crop the image to that box
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img




def transparent_background(
    img: Union[Image.Image, np.ndarray], foreground: np.ndarray
) -> Image.Image:
    """Return a copy of the image with background set to transparent.

    Which pixels belong to the background is specified by
    `foreground`, a 2D array of booleans of the same size as the
    image.
    """
	
    M = np.array(img)
    if M.shape[2]==4:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not foreground[i,j]:
                    M[i,j,3]=0
        return Image.fromarray(M)
    else:
        M[~foreground] = [0, 0, 0]
        return Image.fromarray(np.insert(M, 3, foreground * 255, axis=2))


def contours(img: Image.Image) -> np.ndarray:
    M = np.array(img)
    contour_horizontal = np.abs(M[:-1, 1:] ^ M[:-1, 0:-1])
    contour_vertical = np.abs(M[1:, :-1] ^ M[0:-1, :-1])
    return contour_horizontal + contour_vertical




def surface(img: Image.Image) -> float:
    """Extract the scalar value surface from a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    F = M[:, :, 3] > 0
    return np.count_nonzero(F)


def get_colors(img):
    """Extract various colors from a PIL image."""
    M = np.array(img)
    F = M[:, :, 3] > 0
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    Mx = np.maximum(np.maximum(R, G), B)
    Mn = np.minimum(np.minimum(R, G), B)
    C = Mx - Mn  # Chroma
    D1 = R - (G + B) / 2
    D2 = G - B
    D3 = G - (R + B) / 2
    D4 = B - R
    D5 = B - (G + R) / 2
    D6 = R - G
    # Hue
    # H1 = np.divide(D2, C)
    # H2 = np.divide(D4, C)
    # H3 = np.divide(D6, C)
    # Luminosity
    V = (R + G + B) / 3
    # Saturation
    # S = np.divide(C, V)
    # We avoid divisions so we don't get divisions by 0
    # Now compute the color features
    if not F.any():
        return pd.Series(dtype=float)
    c = np.mean(C[F])
    return pd.Series({
        "R": np.mean(R[F]),
        "G": np.mean(G[F]),
        "B": np.mean(B[F]),
        "M": np.mean(Mx[F]),
        "m": np.mean(Mn[F]),
        "C": np.mean(C[F]),
        "R-(G+B)/2": np.mean(D1[F]),
        "G-B": np.mean(D2[F]),
        "G-(R+B)/2": np.mean(D3[F]),
        "B-R": np.mean(D4[F]),
        "B-(G+R)/2": np.mean(D5[F]),
        "R-G": np.mean(D6[F]),
        "(G-B)/C": np.mean(D2[F]) / c,
        "(B-R)/C": np.mean(D4[F]) / c,
        "(R-G)/C": np.mean(D5[F]) / c,
        "(R+G+B)/3": np.mean(V[F]),
        "C/V": c / np.mean(V[F]),
        })


def color_correlation_filter(
    img: Union[Image.Image, np.ndarray], color: np.ndarray
) -> np.ndarray:
    """Return a grey-level image measuring the correlation of the pixels
    with the given color."""
    M = np.array(img)
    new_M = np.array(
        [
            [np.corrcoef(M[i, j], color)[0, 1] for j in range(len(M))]
            for i in range(len(M))
        ]
    )
    return new_M




def display_images(displayed, grayscale = False):
    num_images = len(displayed)
    cols = 3  
    rows = (num_images // cols) + (num_images % cols > 0)  
    fig = make_subplots(rows=rows, cols=cols)

    
    for idx, img in enumerate(displayed):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        img_array = np.array(img)
        if grayscale:
            img_array = 255*img_array
            img_array = np.stack([img_array] * 3, axis=-1)
        fig.add_trace(go.Image(z=img_array), row=row, col=col)

    
    fig.update_layout(
        height=300 * rows,  # Adjust based on rows
        width=900,  # Fixed width
        showlegend=False
    )

    fig.show()


