"""Track geometry for upstream feature preparation; angles measured from true north."""
import numpy as np


def cyclone_geometry(site_lat,site_lon,tc_lat,tc_lon):
    lat1,lat2=np.deg2rad(site_lat),np.deg2rad(np.asarray(tc_lat))
    dlat=lat2-lat1
    dlon=np.deg2rad((np.asarray(tc_lon)-site_lon+180)%360-180)
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    distance=6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    east=np.sin(dlon)*np.cos(lat2)
    north=np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    bearing=np.arctan2(east,north)
    return dict(tc_distance_km=distance,tc_bearing_sin=np.sin(bearing),tc_bearing_cos=np.cos(bearing))
