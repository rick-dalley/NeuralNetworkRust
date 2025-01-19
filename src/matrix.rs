#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>, // Flat vector for matrix elements
}

impl Matrix {
    // Constructor for a new matrix
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len(), "Data size mismatch");
        Self { rows, cols, data }
    }

    // Zero-initialized matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    // Random-initialized matrix
    pub fn random(rows: usize, cols: usize, range: std::ops::Range<f64>) -> Self {
        use rand::Rng; // Requires `rand` crate
        let mut rng = rand::thread_rng();
        let data = (0..rows * cols)
            .map(|_| rng.gen_range(range.clone()))
            .collect();
        Self { rows, cols, data }
    }

    // Matrix multiplication (dot product)
    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch for dot product");

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }

        result
    }
}