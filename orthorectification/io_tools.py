import gdal
import osr
import numpy as np


def save_raster_as_geotiff(ortho: np.ndarray, ul_lon: float, ul_lat: float, gsd: float, filename: str) -> None:
    io_driver = gdal.GetDriverByName('GTiff')
    ortho_ds = io_driver.Create(filename, ortho.shape[1], ortho.shape[0], 1, gdal.GDT_Byte)
    ortho_ds.SetGeoTransform((ul_lon, gsd, 0, ul_lat, 0, gsd))
    ortho_band = ortho_ds.GetRasterBand(1)
    ortho_band.WriteArray(ortho)
    coordinate_system = osr.SpatialReference()
    coordinate_system.ImportFromEPSG(4326)
    ortho_ds.SetProjection(coordinate_system.ExportToWkt())
    ortho_band.FlushCache()
    # makes sure resources are release...
    ortho_band = None
    ortho_ds = None