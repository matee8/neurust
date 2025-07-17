//! # Generic tensor module.
//!
//! This module provides a generic [`Tensor`] structure that is parameterized by
//! a [backend](crate::backend::Backend).

use core::marker::PhantomData;

use thiserror::Error;

use crate::backend::Backend;

/// Errors that may arise during tensor creation or arithmetic.
#[non_exhaustive]
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationError {
    /// The number of elements provided does not match the given shape.
    #[error(
        r#"the number of elements in the data does not match the number of
        elements required by the shape"#
    )]
    ElementCountMismatch,
    /// Either one of the dimensions, or the product of all dimensions exceeds
    /// [`isize::MAX`].
    #[error(
        "one of the dimensions, or product of all dimensions overflows `isize`"
    )]
    ShapeOverflow,
    /// One of the dimensions is zero.
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
    /// Creates a tensor from a vector and a shape.
    ///
    /// # Errors
    ///
    /// This method will return an [`Err`](OperationError) if the shape is
    /// invalid (see the Errors section on [`Tensor::ones()`]) or if the number
    /// of elements in `data` does not match the number of elements required by
    /// the `shape`. (That is, the number of elements equals to the product of
    /// the dimensions in `shape`.)
    #[inline]
    pub fn from_vec(
        data: Vec<B::Primitive>,
        shape: &[usize],
    ) -> Result<Self, OperationError> {
        Self::validate_shape(shape)?;

        let expected_elements = shape.iter().product::<usize>();

        if data.len() != expected_elements {
            return Err(OperationError::ElementCountMismatch);
        }

        // SAFETY: The shape has been validated and the element count matches
        // the product of the dimensions.
        let inner = unsafe { B::from_vec(data, shape) };

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Returns the number of dimensions of the tensor.
    #[inline]
    #[must_use]
    pub fn ndim(&self) -> usize {
        B::ndim(&self.inner)
    }

    /// Creates a tensor containing only ones.
    ///
    /// # Errors
    ///
    /// See the error notes on [`Tensor::zeros()`].
    #[inline]
    pub fn ones(shape: &[usize]) -> Result<Self, OperationError> {
        Self::validate_shape(shape)?;

        // SAFETY: `shape` does not contain any zeros and product of dimensions
        // does not overflow `isize::MAX`.
        let inner = unsafe { B::ones(shape) };

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
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
    /// This method returns [`Err`](OperationError) if:
    ///
    /// - one of the dimensions in the parameter value is 0,
    /// - any of the axis lengths overflows [`isize::MAX`],
    /// - the product of axis lengths overflows `isize::MAX`.
    #[inline]
    pub fn zeros(shape: &[usize]) -> Result<Self, OperationError> {
        Self::validate_shape(shape)?;

        // SAFETY: `shape` does not contain any zeros, no dimensions overflow
        // `isize`, the product of dimensions does not overflow `isize::MAX`.
        let inner = unsafe { B::zeros(shape) };

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Validates shape for tensor creation.
    ///
    /// This private helper function ensures that all preconditions for creating
    /// a tensor from a shape are met. It checks for:
    /// - zero-sized dimensions,
    /// - any of the dimensions overflowing `isize`,
    /// - overflow when calculating the total number of elements.
    fn validate_shape(shape: &[usize]) -> Result<(), OperationError> {
        if shape.contains(&0) {
            return Err(OperationError::ZeroDim);
        }

        let _: isize = shape.iter().try_fold(1_isize, |product, &dim| {
            let dim_isize = isize::try_from(dim)
                .map_err(|_| OperationError::ShapeOverflow)?;
            product
                .checked_mul(dim_isize)
                .ok_or(OperationError::ShapeOverflow)
        })?;

        Ok(())
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
        value: <MockBackend as Backend>::Primitive,
    }

    impl Backend for MockBackend {
        type Primitive = f32;
        type Tensor = MockTensor;

        unsafe fn from_vec(
            data: Vec<Self::Primitive>,
            shape: &[usize],
        ) -> Self::Tensor {
            let value = data.first().copied().unwrap_or_default();

            Self::Tensor {
                shape: shape.to_owned(),
                value,
            }
        }

        fn ndim(tensor: &Self::Tensor) -> usize {
            tensor.shape.len()
        }

        unsafe fn ones(shape: &[usize]) -> Self::Tensor {
            Self::Tensor {
                shape: shape.to_owned(),
                value: 1.0,
            }
        }

        fn shape(tensor: &Self::Tensor) -> &[usize] {
            &tensor.shape
        }

        unsafe fn zeros(shape: &[usize]) -> Self::Tensor {
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
    fn test_tensor_creation_zeros_fails_on_overflow() {
        let isize_max = usize::try_from(isize::MAX).unwrap();
        let tensor = Tensor::<MockBackend>::zeros(&[isize_max, 2]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ShapeOverflow);
    }

    #[test]
    fn test_tensor_creation_ones_works() {
        let tensor = Tensor::<MockBackend>::ones(&[2, 3]);

        assert!(tensor.is_ok());
    }

    #[test]
    fn test_tensor_creation_ones_has_right_shape() {
        let shape = &[2, 3];
        let tensor = Tensor::<MockBackend>::ones(shape).unwrap();

        assert_eq!(tensor.inner.shape, shape);
    }

    #[test]
    fn test_tensor_creation_ones_has_right_values() {
        let tensor = Tensor::<MockBackend>::ones(&[2, 3]).unwrap();

        assert_eq!(tensor.inner.value, 1.0);
    }

    #[test]
    fn test_tensor_validation_fails_on_zero_dimensions() {
        let result = Tensor::<MockBackend>::validate_shape(&[2, 0]);

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(err, OperationError::ZeroDim);
    }

    #[test]
    fn test_tensor_creation_ones_fails_on_zero_dimension() {
        let tensor = Tensor::<MockBackend>::ones(&[2, 0]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ZeroDim);
    }

    #[test]
    fn test_tensor_creation_ones_fails_on_overflow() {
        let isize_max = usize::try_from(isize::MAX).unwrap();
        let tensor = Tensor::<MockBackend>::ones(&[isize_max, 2]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ShapeOverflow);
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

    #[test]
    fn test_tensor_validation_fails_on_too_large_usize() {
        let usize_max = usize::MAX;
        let result = Tensor::<MockBackend>::validate_shape(&[usize_max, 1]);

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(err, OperationError::ShapeOverflow);
    }

    #[test]
    fn test_tensor_validation_fails_on_overflow() {
        let isize_max = usize::try_from(isize::MAX).unwrap();
        let result = Tensor::<MockBackend>::validate_shape(&[isize_max, 2]);

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(err, OperationError::ShapeOverflow);
    }

    #[test]
    fn test_tensor_validation_succeeds_on_correct_shape() {
        let result = Tensor::<MockBackend>::validate_shape(&[2, 3]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_creation_from_vec_succeeds() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = &[2, 3];
        let tensor = Tensor::<MockBackend>::from_vec(data, shape);

        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();

        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn test_tensor_creation_from_vec_fails_on_too_few_elements() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::<MockBackend>::from_vec(data, &[2, 3]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ElementCountMismatch);
    }

    #[test]
    fn test_tensor_creation_from_vec_fails_on_too_many_elements() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::<MockBackend>::from_vec(data, &[1, 2]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ElementCountMismatch);
    }

    #[test]
    fn test_tensor_creation_from_vec_fails_on_invalid_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::<MockBackend>::from_vec(data, &[2, 0]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, OperationError::ZeroDim);
    }
}
