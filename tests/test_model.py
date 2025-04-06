import unittest
import numpy as np
from A3_jang import LogisticRegression  # ปรับชื่อไฟล์ model.py ให้ตรงกับของคุณ

class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        # ตั้งค่าข้อมูลจำลอง
        self.X_dummy = np.random.rand(10, 6)  # 10 ตัวอย่าง, 6 features
        self.Y_dummy = np.zeros((10, 4))
        self.Y_dummy[np.arange(10), np.random.randint(0, 4, 10)] = 1  # one-hot encoded
        
        # สร้างโมเดล
        self.model = LogisticRegression(k=4, n=6, method="batch", max_iter=10)

    def test_input_shape(self):
        try:
            self.model.fit(self.X_dummy, self.Y_dummy)
        except Exception as e:
            self.fail(f"Model raised an error with valid input: {e}")

    def test_output_shape(self):
        self.model.fit(self.X_dummy, self.Y_dummy)
        pred = self.model.predict(self.X_dummy)
        self.assertEqual(pred.shape, (10,), "Output shape should be (num_samples,)")

if __name__ == '__main__':
    unittest.main()
