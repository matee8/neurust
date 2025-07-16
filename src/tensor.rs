//! # Generic tensor module.
//!
//! This module provides a generic `Tensor` structure that is parameterized by
//! a backend.

use core::marker::PhantomData;

use thiserror::Error;

use crate::backend::Backend;

/// Errors that may arise during tensor creation or arithmetic.
#[non_exhaustive]
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationError {
    #[error("dimensions must be non-zero")]
    ZeroDim,
}

/// Generic, backend-agnostic n-dimensional tensor.
///
/// `Tensor` is a thin wrapper around a backend-specific tensor implementation.
/// It only performs invariants checks at runtime and delegates the actual
/// maths to the backend.
#[derive(Debug)]
pub struct Tensor<B>
where
    B: Backend,
{
    _marker: PhantomData<B>,
    inner: B::Tensor,
}

impl<B> Tensor<B>
where
    B: Backend,
{
    /// Returns the number of dimensions of the tensor.
    #[inline]
    #[must_use]
    pub fn ndim(&self) -> usize {
        B::ndim(&self.inner)
    }

    /// Returns the shape of the tensor as a slice of dimensions.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        B::shape(&self.inner)
    }

    /// Create a tensor containing only zeros.
    ///
    /// # Errors
    ///
    /// This method returns `Err` if, and only if, one of the dimensions in the
    /// parameter value is 0.
    #[inline]
    pub fn zeros(shape: &[usize]) -> Result<Self, OperationError> {
        if shape.contains(&0) {
            return Err(OperationError::ZeroDim);
        }

        Ok(Self {
            inner: B::zeros(shape),
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::tensor::{Backend, OperationError};

    #[derive(Debug)]
    struct MockBackend;

    #[derive(Debug)]
    struct MockTensor {
        shape: Vec<usize>,
        value: f32,
    }

    impl Backend for MockBackend {
        type Tensor = MockTensor;

        fn ndim(tensor: &Self::Tensor) -> usize {
            tensor.shape.len()
        }

        fn shape(tensor: &Self::Tensor) -> &[usize] {
            &tensor.shape
        }

        fn zeros(shape: &[usize]) -> Self::Tensor {
            Self::Tensor {
                shape: shape.to_owned(),
                value: 0.0,
            }
        }
    }

    #[test]
    fn test_tensor_creation_zeros_works() {
        let tensor = Tensor::<MockBackend>::zeros(&[2, 3]);

        assert!(tensor.is_ok());
    }

    #[test]
    fn test_tensor_creation_zeros_has_right_shape() {
        let shape = &[2, 3];
        let tensor = Tensor::<MockBackend>::zeros(shape).unwrap();

        assert_eq!(tensor.inner.shape, shape);
    }

    #[test]
    fn test_tensor_creation_zeros_has_right_value() {
        let tensor = Tensor::<MockBackend>::zeros(&[2, 3]).unwrap();

        assert_eq!(tensor.inner.value, 0.0);
    }

    #[test]
    fn test_tensor_creation_zeros_fails_on_zero_dimension() {
        let tensor = Tensor::<MockBackend>::zeros(&[2, 0]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ZeroDim);
    }

    #[test]
    fn test_tensor_shape_is_correct() {
        let shape = &[2, 3];
        let tensor = Tensor::<MockBackend>::zeros(shape).unwrap();

        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn test_tensor_ndim_is_correct() {
        let tensor = Tensor::<MockBackend>::zeros(&[2, 3]).unwrap();

        assert_eq!(tensor.ndim(), 2);
    }
}
