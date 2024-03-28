import unittest
from .PyAdfTestCase import PyAdfTestCase

import numpy as np
from ...kf.kf import kffile


class KFFileTests(PyAdfTestCase):
    """
    Unit tests for the KFFile class.
    """

    def setUp(self):
        import tempfile
        import os.path

        with tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False) as tf:
            self.kfPath = tf.name
            tf.file.close()
            os.remove(tf.name)

    def tearDown(self):
        import os
        if os.path.isfile(self.kfPath):
            os.remove(self.kfPath)

    def testLogicals(self):
        with kffile(self.kfPath) as f:
            lint = 1
            f.writelogicals('Logicals', 'scalar true', lint)
            lout = f.read('Logicals', 'scalar true')
            self.assertEqual(lout[0], lint)
            self.assertEqual(len(lout), 1)

            linf = 0
            f.writelogicals('Logicals', 'scalar false', linf)
            lout = f.read('Logicals', 'scalar false')
            self.assertEqual(lout[0], linf)
            self.assertEqual(len(lout), 1)

            lin = np.array([0, 1, 0, 1])
            f.writelogicals('Logicals', 'list', lin)
            lout = f.read('Logicals', 'list')
            np.testing.assert_equal(lout, lin)

    def testReals(self):
        with kffile(self.kfPath) as f:
            rin_scalar = 3.14
            f.writereals('Reals', 'scalar', rin_scalar)
            rout = f.read('Reals', 'scalar')
            self.assertEqual(rin_scalar, rout[0])

            rin_list = [0.0, 3.14, -1.0e-16, 3e24]
            f.writereals('Reals', 'list', rin_list)
            rout = f.read('Reals', 'list')
            np.testing.assert_equal(rin_list, rout)

    def testChars(self):
        with kffile(self.kfPath) as f:
            cin_long = "This is a long character string to test the pykf stuff, will it work or will it not? " + \
                       "The string certainly is long."
            f.writechars('String', 'scalar', cin_long)
            cout = f.read('String', 'scalar')

            self.assertEqual(cin_long, cout[0])

            cin_list = ["String 1", "String 2", "Yet another String"]
            f.writechars('String', 'list', cin_list)
            cout = f.read('String', 'list')
            np.testing.assert_equal(cin_list, cout)

    def testInts(self):
        with kffile(self.kfPath) as f:
            iin_scalar = 3
            f.writeints('Ints', 'scalar', iin_scalar)
            iout = f.read('Ints', 'scalar')
            self.assertEqual(iin_scalar, iout[0])

            iin_list = [0, 1, 2, 3, 4, 5, -123]
            f.writereals('Ints', 'list', iin_list)
            iout = f.read('Ints', 'list')
            np.testing.assert_equal(iin_list, iout)

    def testNone(self):
        with kffile(self.kfPath) as f:
            f.writeints('Bla', 'BlaBla', 1)
            with self.assertRaises(KeyError):
                f.read('Blurb', 'jojo')

    def testCasesensitive(self):
        with kffile(self.kfPath) as f:
            i = 0
            f.writeints('Names', 'Aap', i)
            with self.assertRaises(KeyError):
                f.read('Names', 'aap')
            with self.assertRaises(KeyError):
                f.read('names', 'Aap')
            ii = f.read('Names', 'Aap')
            self.assertEqual(ii[0], i)

    def testFileExists(self):
        import os
        with kffile(self.kfPath) as f:
            f.writechars('Test', 'string', "Hello World")

        self.assertTrue(os.path.isfile(self.kfPath))


if __name__ == "__main__":
    allTestsSuite = unittest.TestSuite([unittest.makeSuite(KFFileTests, 'test')])
    runner = unittest.TextTestRunner()
    runner.run(allTestsSuite)
