//! # Generic tensor module.
//!
//! This module provides a generic `Tensor` structure that is parameterized by
//! a backend. The `Backend` trait defines the complete contract for all
//! linear algebra operations, data management, and tensor creation.
//!
//! The default backend is `ndarray` and can be swapped out using crate
//! feature flags.

use core::marker::PhantomData;

use thiserror::Error;

/// A trait that defines the contract for tensor operations that every
/// backend must fulfill.
///
/// The `Backend` trait is the core abstraction of the tensor module. It
/// provides a generic interface for tensor creation, manipulation, and
/// computation. By implementing this trait, different computation libraries
/// (like `ndarray` or `nalgebra`) can be used as the underlying engine for
/// tensor operations. This allows for flexibility and performance tuning
/// by switching backends through feature flags. All functions are pure.
pub trait Backend {
    /// The concrete tensor representation provided by the backend.
    type Tensor;

    /// Creates a tensor with all elements set to zero, with the given shape.
    fn zeros(shape: &[usize]) -> Self::Tensor;
}

/// Errors that may arise during tensor creation or arithmetic.
#[non_exhaustive]
#[derive(Error, Debug, Clone, Copy)]
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
    /// Create a tensor containing only zeros.
    ///
    /// # Errors
    ///
    /// This method returns `Err` if, and only if, one of the dimensions in the
    /// parameter value is 0.
    #[inline]
    pub fn zeros(shape: &[usize]) -> Result<Self, OperationError> {
        if shape.iter().any(|&dim| dim == 0) {
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
    use crate::tensor::Backend;

    struct MockBackend;

    struct MockTensor {
        shape: Vec<usize>,
        value: f32,
    }

    impl Backend for MockBackend {
        type Tensor = MockTensor;

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
}
