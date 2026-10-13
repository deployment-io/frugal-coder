//! Contiguous row-major f32 tensors of rank 1 or 2, with nonzero dimensions.
//!
//! All inputs (including RMS weights) must be finite. Operations use f64
//! intermediates/accumulation, then round to f32; nonfinite outputs are errors.
//! RMSNorm and softmax operate independently along the last dimension. No
//! broadcasting, strides, autograd, model-specific normalization, or SIMD.
use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorError(pub String);

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for TensorError {}

type Result<T> = std::result::Result<T, TensorError>;

/// Validate rank, element count, and Rust allocation size without allocating.
pub fn checked_elements(shape: &[usize]) -> Result<usize> {
    if !(1..=2).contains(&shape.len()) {
        return Err(TensorError("only rank 1 or 2 is supported".into()));
    }
    if shape.contains(&0) {
        return Err(TensorError("empty dimensions are unsupported".into()));
    }
    let count = shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| TensorError("shape multiplication overflow".into()))
    })?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| TensorError("allocation byte-size overflow".into()))?;
    if bytes > isize::MAX as usize {
        return Err(TensorError("allocation exceeds isize::MAX bytes".into()));
    }
    Ok(count)
}

fn finite(data: &[f32], context: &str) -> Result<()> {
    if data.iter().any(|v| !v.is_finite()) {
        return Err(TensorError(format!(
            "{context} must contain only finite values"
        )));
    }
    Ok(())
}

fn rounded(value: f64) -> Result<f32> {
    let result = value as f32;
    if !result.is_finite() {
        return Err(TensorError(
            "output is not representable as finite f32".into(),
        ));
    }
    Ok(result)
}

fn output(shape: &[usize]) -> Result<Vec<f32>> {
    let count = checked_elements(shape)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|e| TensorError(format!("cannot allocate tensor: {e}")))?;
    values.resize(count, 0.0);
    Ok(values)
}

/// Construction establishes shape, storage, and finite-value invariants.
/// Storage and shape cannot be mutated through the public API.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected = checked_elements(&shape)?;
        if data.len() != expected {
            return Err(TensorError(format!(
                "shape requires {expected} elements, storage has {}",
                data.len()
            )));
        }
        finite(&data, "tensor storage")?;
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// A[M,K] @ B[K,N] -> C[M,N]. Dot products accumulate in f64.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(TensorError("matmul requires two rank-2 tensors".into()));
        }
        let (m, k, n) = (self.shape[0], self.shape[1], rhs.shape[1]);
        if k != rhs.shape[0] {
            return Err(TensorError("matmul inner dimensions do not match".into()));
        }
        let shape = vec![m, n];
        let mut data = output(&shape)?;
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f64;
                for inner in 0..k {
                    sum += f64::from(self.data[row * k + inner])
                        * f64::from(rhs.data[inner * n + col]);
                }
                data[row * n + col] = rounded(sum)?;
            }
        }
        Ok(Self { shape, data })
    }

    /// Row RMSNorm: x / sqrt(mean(x*x) + epsilon) * weight.
    /// No mean subtraction; weights are direct scales, not zero-centered offsets.
    pub fn rms_norm(&self, weight: &[f32], epsilon: f32) -> Result<Self> {
        let width = self.shape[self.shape.len() - 1];
        if weight.len() != width {
            return Err(TensorError(
                "RMSNorm weight length must match last dimension".into(),
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(TensorError(
                "epsilon must be finite and strictly positive".into(),
            ));
        }
        finite(weight, "RMSNorm weights")?;
        let mut data = output(&self.shape)?;
        for (src, dst) in self
            .data
            .chunks_exact(width)
            .zip(data.chunks_exact_mut(width))
        {
            let sum: f64 = src.iter().map(|&v| f64::from(v).powi(2)).sum();
            let divisor = (sum / width as f64 + f64::from(epsilon)).sqrt();
            for ((&x, &w), y) in src.iter().zip(weight).zip(dst) {
                *y = rounded(f64::from(x) / divisor * f64::from(w))?;
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Stable row softmax for finite inputs; underflow to zero is allowed.
    pub fn softmax(&self) -> Result<Self> {
        let width = self.shape[self.shape.len() - 1];
        let mut data = output(&self.shape)?;
        for (src, dst) in self
            .data
            .chunks_exact(width)
            .zip(data.chunks_exact_mut(width))
        {
            let max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // Subtract in f64 so opposite-sign finite f32 extremes cannot overflow.
            let sum: f64 = src
                .iter()
                .map(|&v| (f64::from(v) - f64::from(max)).exp())
                .sum();
            for (&x, y) in src.iter().zip(dst) {
                *y = rounded((f64::from(x) - f64::from(max)).exp() / sum)?;
            }
        }
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }

    /// Elementwise SiLU, using a sign-dependent sigmoid to avoid exp overflow.
    pub fn silu(&self) -> Result<Self> {
        let mut data = output(&self.shape)?;
        for (&x, y) in self.data.iter().zip(&mut data) {
            let x = f64::from(x);
            let sigmoid = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp = x.exp();
                exp / (1.0 + exp)
            };
            *y = rounded(x * sigmoid)?;
        }
        Ok(Self {
            shape: self.shape.clone(),
            data,
        })
    }
}
