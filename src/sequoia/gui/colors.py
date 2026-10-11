from PySide6.QtGui import QColor


def wavelength_color(wavelength_m: float) -> QColor:
    """Approximate display colour of a wavelength; UV and IR fade to violet and dark red."""
    nm = wavelength_m * 1e9
    if nm < 380:
        r, g, b, f = 0.5, 0.2, 0.8, 1.0
    elif nm < 440:
        r, g, b, f = (440 - nm) / 60, 0.0, 1.0, 0.3 + 0.7 * (nm - 380) / 60
    elif nm < 490:
        r, g, b, f = 0.0, (nm - 440) / 50, 1.0, 1.0
    elif nm < 510:
        r, g, b, f = 0.0, 1.0, (510 - nm) / 20, 1.0
    elif nm < 580:
        r, g, b, f = (nm - 510) / 70, 1.0, 0.0, 1.0
    elif nm < 645:
        r, g, b, f = 1.0, (645 - nm) / 65, 0.0, 1.0
    elif nm <= 780:
        r, g, b, f = 1.0, 0.0, 0.0, 0.3 + 0.7 * (780 - nm) / 135
    else:
        r, g, b, f = 0.55, 0.0, 0.0, 1.0
    # darken so that yellow-green stays visible on a light canvas
    return QColor.fromRgbF(r * f * 0.85, g * f * 0.75, b * f * 0.9)
