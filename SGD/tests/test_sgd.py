"""Independent checks for normalization, non-smooth math, leakage and RNG bugs."""
import unittest
import numpy as np
import sgd

class MathTests(unittest.TestCase):
    def test_objective_and_unregularized_bias(self):
        X=np.array([[1.,2.],[-1.,0.]])
        Y=np.array([[1.],[-1.]])
        W=np.array([[.2],[.1]]); b=np.array([.3])
        # margins .7,-.1 -> mean hinge .7, penalty .005
        self.assertAlmostEqual(sgd.objective(X,Y,W,b,.2),.705)
        g,h=sgd.subgradient(X,Y,W,b,.2)
        np.testing.assert_allclose(g,[[-.96],[-.98]])
        np.testing.assert_allclose(h,[0.])
        self.assertAlmostEqual(sgd.objective(np.zeros((2,2)),Y,W,np.array([5.]),.2),3.005)

    def test_gradient_matches_finite_difference_per_head(self):
        rng=np.random.default_rng(17)
        X=rng.normal(size=(7,3)); Y=rng.choice([-1.,1.],size=(7,2))
        W=rng.normal(size=(3,2))*.1; b=np.array([.03,-.07])
        g,h=sgd.subgradient(X,Y,W,b,.13); eps=1e-6
        # Objective is a mean over heads, gradients update independent F_k.
        for j in range(2):
            for i in range(3):
                A=W.copy();B=W.copy();A[i,j]+=eps;B[i,j]-=eps
                fd=(sgd.objective(X,Y,A,b,.13)-sgd.objective(X,Y,B,b,.13))/(2*eps)*2
                self.assertAlmostEqual(fd,g[i,j],places=7)
            a=b.copy();c=b.copy();a[j]+=eps;c[j]-=eps
            fd=(sgd.objective(X,Y,W,a,.13)-sgd.objective(X,Y,W,c,.13))/(2*eps)*2
            self.assertAlmostEqual(fd,h[j],places=7)

    def test_margin_kink_uses_zero_hinge_subgradient(self):
        g,h=sgd.subgradient(np.array([[1.]]),np.array([[1.]]),np.array([[1.]]),np.array([0.]),.2)
        np.testing.assert_allclose(g,[[.2]]);np.testing.assert_allclose(h,[0.])

    def test_short_batch_uses_its_actual_size(self):
        g,h=sgd.subgradient(np.array([[2.,-3.]]),np.array([[1.,-1.]]),np.zeros((2,2)),np.zeros(2),.1)
        np.testing.assert_allclose(g,[[-2.,2.],[3.,-3.]])
        np.testing.assert_allclose(h,[-1.,1.])

    def test_schedule_is_measured_in_examples_seen(self):
        self.assertAlmostEqual(sgd.learning_rate(.2,0,40,10),.2)
        self.assertAlmostEqual(sgd.learning_rate(.2,.5,30,10),.1)
        self.assertAlmostEqual(sgd.learning_rate(.2,1,10,10),.1)

class PipelineTests(unittest.TestCase):
    def test_split_disjoint_complete_and_scaler_train_only(self):
        data=sgd.prepare_digits(2026)
        ids=data['indices'];sets=[set(ids[k]) for k in ('train','validation','test')]
        self.assertEqual([len(v) for v in sets],[1077,360,360])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),1797)
        raw=data['raw_X'][ids['train']]
        np.testing.assert_allclose(data['scaler'].mean_,raw.mean(axis=0),atol=1e-14)
        np.testing.assert_allclose(data['X_train'].mean(axis=0),0,atol=1e-14)
        self.assertTrue(np.array_equal(data['classes'],np.arange(10)))

    def test_every_epoch_covers_every_sample_once(self):
        rng=np.random.default_rng(4)
        one=list(sgd.epoch_batches(7,3,rng));two=list(sgd.epoch_batches(7,3,rng))
        self.assertEqual([len(v) for v in one],[3,3,1])
        np.testing.assert_array_equal(np.sort(np.concatenate(one)),np.arange(7))
        self.assertFalse(np.array_equal(np.concatenate(one),np.concatenate(two)))

    def test_reproducible_updates_and_real_progress(self):
        X=np.array([[-2.],[-1.],[1.],[2.]])
        y=np.array([0,0,1,1]);classes=np.array([0,1])
        cfg=dict(lam=.01,eta0=.1,power=.5,batch_size=3,epochs=15)
        a=sgd.fit(X,y,X,y,classes,cfg,13)
        b=sgd.fit(X,y,X,y,classes,cfg,13)
        np.testing.assert_array_equal(a['W'],b['W'])
        np.testing.assert_array_equal(a['b'],b['b'])
        self.assertEqual(a['history'][-1]['updates'],30)
        self.assertEqual(a['history'][-1]['samples_seen'],60)
        self.assertLess(a['history'][-1]['train_objective'],.1)
        self.assertEqual(a['history'][-1]['validation_accuracy'],1.)
        for h in a['history']:self.assertGreaterEqual(h['fit_wall_seconds'],h['optimizer_seconds'])

if __name__=='__main__':unittest.main()
