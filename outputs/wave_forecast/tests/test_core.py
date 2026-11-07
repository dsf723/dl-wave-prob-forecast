import tempfile
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from wavecast.models import TiDE,Transformer,BernsteinHead,crps_loss
from wavecast.metrics import score,paired_bootstrap
from wavecast.data import read_tables,build_samples,feature_names,diagnostic_labels,Scaler
from wavecast.demo import generate
from wavecast.convert_gefs import convert
from wavecast.experiment import mixture_quantiles
from wavecast.geo import cyclone_geometry
from wavecast.pseudo import generate as generate_pseudo


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.data=generate(Path(cls.tmp.name)/"data")
        cls.cfg=json.loads((Path(__file__).resolve().parents[1]/"config.json").read_text(encoding="utf-8"))
        cls.cfg.update(data_dir=str(cls.data),mode="synthetic",site_id="DEMO",train_years=[2018,2019],
                       val_years=[2020],test_years=[2021],decision_delay_hours=0,hidden=16,layers=1,heads=2)
        cls.obs,cls.fc,_=read_tables(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_probability_heads_and_gradients(self):
        torch.set_num_threads(2)
        for klass in [TiDE,Transformer]:
            m=klass(8,56,112,self.cfg)
            q=m(torch.randn(2,56,8),torch.randn(2,112,8))
            self.assertEqual(tuple(q.shape),(2,112,99))
            self.assertTrue((q>=0).all())
            self.assertTrue((q.diff(dim=-1)>=-1e-6).all())
            loss=crps_loss(q,torch.rand(2,112),m.head.tau)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_crps_point_distribution_is_absolute_error(self):
        q=np.repeat(np.array([[1.],[4.]]),99,axis=1)
        np.testing.assert_allclose(score(q,np.array([2.,2.])),[1.,2.],rtol=1e-6)

    def test_future_truth_cannot_change_inputs_or_route(self):
        first=self.fc.issue_time.min()
        fc=self.fc[self.fc.issue_time==first]
        names=feature_names(self.cfg,"extreme")
        a,_=build_samples(self.obs,fc,self.cfg,names)
        altered=self.obs.copy()
        ix=altered.index>first
        altered.loc[ix,"hs"]+=30
        altered.loc[ix,"tc_distance_km"]=0
        b,_=build_samples(altered,fc,self.cfg,names)
        np.testing.assert_allclose(a["train"][0]["future"],b["train"][0]["future"],equal_nan=True)
        np.testing.assert_allclose(a["train"][0]["past"],b["train"][0]["past"],equal_nan=True)
        self.assertEqual([m["routing_regime"] for m in a["train"][0]["metadata"]],
                         [m["routing_regime"] for m in b["train"][0]["metadata"]])
        self.assertFalse(np.allclose(a["train"][0]["y"],b["train"][0]["y"]))

    def test_unpublished_forecast_is_excluded(self):
        fc=self.fc[self.fc.issue_time==self.fc.issue_time.min()].copy()
        fc["available_time"]=fc.issue_time+pd.Timedelta(hours=1)
        a,skip=build_samples(self.obs,fc,self.cfg,feature_names(self.cfg,"base"))
        self.assertFalse(a["train"])
        self.assertEqual(skip["forecast_unavailable"],1)

    def test_late_history_excluded_in_real_mode(self):
        cfg=dict(self.cfg,mode="real")
        first=self.fc.issue_time.min(); fc=self.fc[self.fc.issue_time==first]
        obs=self.obs.copy(); obs["available_time"]=obs.available_time+pd.Timedelta(days=5)
        samples,_=build_samples(obs,fc,cfg,feature_names(cfg,"base"))
        self.assertFalse(samples["train"])

    def test_split_boundaries_and_event_separation(self):
        samples,_=build_samples(self.obs,self.fc,self.cfg,feature_names(self.cfg,"base"))
        for split,rows in samples.items():
            for row in rows:
                self.assertTrue(all(m["valid_time"].year in self.cfg[split+"_years"] for m in row["metadata"]))
        obs=self.obs.copy()
        obs.loc[(obs.event_id!=""),"event_id"]="same_event_all_years"
        with self.assertRaisesRegex(ValueError,"event crosses"):
            build_samples(obs,self.fc,self.cfg,feature_names(self.cfg,"base"))

    def test_scaler_training_only_and_missingness(self):
        s=Scaler().fit([dict(past=np.array([[1,np.nan],[3,np.nan]]),future=np.array([[2,np.nan]]))])
        np.testing.assert_allclose(s.mean,[2,0])
        self.assertEqual(s.all_missing,[1])
        x=s.transform(np.array([[np.nan,7]]))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[0,2:],[0,1])

    def test_unknown_weather_never_assumed_calm(self):
        row=pd.Series(dict(hs=.2,u10=0,v10=0,temp_change_24h=0,tc_distance_km=np.nan))
        self.assertEqual(diagnostic_labels(row,self.cfg)[0],"unknown")
        row.tc_distance_km=20000
        self.assertEqual(diagnostic_labels(row,self.cfg)[0],"calm")
        row.u10=0; row.v10=-15; row.temp_change_24h=-10
        self.assertEqual(diagnostic_labels(row,self.cfg)[0],"cold_surge")
        row.tc_distance_km=100
        self.assertEqual(diagnostic_labels(row,self.cfg)[0],"typhoon_and_cold")

    def test_paired_cluster_bootstrap(self):
        a=pd.DataFrame(dict(site_id=["A"]*12,issue_time=["x"]*12,valid_time=range(12),
                            block_id=["storm1"]*6+["storm2"]*6,crps=[1.]*12))
        b=a.copy(); b["crps"]=2.
        r=paired_bootstrap(a,b,100)
        self.assertEqual(r["n_blocks"],2)
        self.assertEqual(r["ci_high"],-1)

    def test_gefs_circle_and_availability(self):
        path=Path(self.tmp.name)/"members.csv"
        pd.DataFrame(dict(member=["a","b","a","b"],valid_time=["2020-01-01"]*2+["2020-01-01 06:00"]*2,
                          swh=[1,2,2,3],mwd=[350,10,350,10],u=[0]*4,v=[-2]*4)).to_csv(path,index=False)
        out=convert(path,"2020-01-01T00:00Z","2020-01-01T08:00Z","A",min_members=2)
        self.assertEqual(len(out),3)
        self.assertTrue(np.all(np.minimum(out.mwd,360-out.mwd)<1e-5))
        self.assertTrue((out.available_time==pd.Timestamp("2020-01-01T08:00Z")).all())
        self.assertNotIn("te",out)

    def test_mixture_quantiles_ordered_and_nonnegative(self):
        a=np.linspace(.1,2,99)[None,:]; b=np.linspace(4,8,99)[None,:]
        q=mixture_quantiles(a,b)
        self.assertTrue((np.diff(q)>=0).all())
        self.assertGreater(q[0,90],4)

    def test_track_distance_wraps_dateline(self):
        g=cyclone_geometry(0,179,0,-179)
        self.assertAlmostEqual(float(g["tc_distance_km"]),222.39016,places=3)
        self.assertAlmostEqual(float(g["tc_bearing_sin"]),1,places=6)

    def test_pseudo_inputs_have_no_future_best_track(self):
        source=Path(self.tmp.name)/"short_observations.csv"
        self.obs.iloc[:31*8].reset_index().to_csv(source,index=False)
        root=Path(self.tmp.name)/"pseudo_test"
        generate_pseudo(source,root,error_scale=1,origin_stride_days=7)
        f=pd.read_csv(root/"forecasts.csv")
        self.assertNotIn("event_id",f)
        self.assertNotIn("tc_distance_km",f)
        self.assertNotIn("label_tc_distance_km",f)
        cfg=dict(self.cfg,data_dir=str(root),mode="pseudo_forecast")
        obs,forecast,_=read_tables(cfg)
        matched=forecast.set_index("valid_time").hs-obs.hs.reindex(forecast.valid_time).to_numpy()
        self.assertGreater(matched.abs().mean(),0)


if __name__=="__main__":
    unittest.main()
