import unittest
from pathlib import Path
import tempfile
import numpy as np
import sgd

class CreditTests(unittest.TestCase):
    def test_numeric_cleaning_does_not_learn_from_other_customers(self):
        self.assertEqual(sgd.parse_number('12_'),12.)
        self.assertTrue(np.isnan(sgd.parse_number('_______')))
        self.assertEqual(sgd.parse_history('3 Years and 2 Months'),38.)
        self.assertTrue(np.isnan(sgd.parse_history('NA')))
        train=np.array([[1.,np.nan],[3.,4.],[5.,6.]])
        prep=sgd.NumericPreprocessor().fit(train)
        mean=prep.mean.copy();median=prep.median.copy()
        transformed=prep.transform(np.array([[1e9,np.nan]]))
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(prep.mean,mean)
        np.testing.assert_array_equal(prep.median,median)
        np.testing.assert_allclose(median,[3.,5.])

    def test_customer_group_split_is_disjoint(self):
        groups=np.repeat(np.arange(30),8);y=np.tile(np.arange(8)%3,30)
        ids=sgd.credit_split(y,groups,2026)
        sets=[set(groups[ids[k]]) for k in ('train','validation','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),30)
        self.assertEqual(sum(len(v) for v in ids.values()),240)

if __name__=='__main__':unittest.main()
