//! # Generic tensor module.
//!
//! This module provides a generic [`TensorBase`] structure that is
//! parameterized by a [backend](crate::backend::Backend), and convenient type
//! aliases for the most common use cases.

use core::{marker::PhantomData, ops::Add};

use thiserror::Error;

use crate::backend::{Backend, SelectedBackend};

/// A tensor with a backend selected at compile-time, generic over the element
/// type.
///
/// This is the primary, user-facing tensor type. It utilizes the backend chosen
/// via crate features, providing a simple and direct API for most use cases.
/// However, for operations with different backends simultaneously, see
/// [`TensorBase`].
pub type Tensor<T> = TensorBase<SelectedBackend<T>>;

/// A `Tensor` with an `f32` primitive type.
///
/// This is a convenience type alias for the most common tensor configuration
/// used in machine learning.
pub type FloatTensor = Tensor<f32>;

/// An error related to the shape of a tensor during its creation.
///
/// This error is returned by constructors like [`TensorBase::zeros()`] when the
/// provided shape is invalid or its properties exceed system limits.
#[non_exhaustive]
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
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

/// An error that occurs during the tensor operations, when the two given
/// tensors are incompatible with eachother.
///
/// This error is returned by operations like [`TensorBase::add()`].
#[non_exhaustive]
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompatibleTensorsError {
    /// The shapes of the input tensors are not compatible for the operation.
    #[error("incompatible tensor shapes for operation")]
    ShapeMismatch,
}

/// Generic, backend-agnostic n-dimensional tensor.
///
/// `TensorBase` is a thin wrapper around a backend-specific tensor
/// implementation. It only performs invariants checks at runtime and delegates
/// the actual maths to the backend.
///
/// For most applications, the [`Tensor`] type alias is more convenient.
#[derive(Debug)]
pub struct TensorBase<B>
where
    B: Backend,
{
    _marker: PhantomData<B>,
    inner: B::Tensor,
}

impl<B> TensorBase<B>
where
    B: Backend,
{
    /// Performs element-wise addition between two tensors.
    ///
    /// # Errors
    ///
    /// Returns a [`IncompatibleTensorsError::ShapeMismatch`] if the tensors
    /// do not have the same shape.
    #[inline]
    pub fn checked_add(
        &self,
        other: &Self,
    ) -> Result<Self, IncompatibleTensorsError> {
        self.checked_binary_op(other, |lhs, rhs| {
            // SAFETY: The shapes are guaranteed to be the same.
            unsafe { B::add(lhs, rhs) }
        })
    }

    /// Performs element-wise subtraction between two tensors.
    ///
    /// # Errors
    ///
    /// Returns a [`IncompatibleTensorsError::ShapeMismatch`] if the tensors
    /// do not have the same shape.
    #[inline]
    pub fn checked_sub(
        &self,
        other: &Self,
    ) -> Result<Self, IncompatibleTensorsError> {
        self.checked_binary_op(other, |lhs, rhs| {
            // SAFETY: The shapes are guaranteed to be the same.
            unsafe { B::sub(lhs, rhs) }
        })
    }

    /// Creates a tensor from a vector and a shape.
    ///
    /// # Errors
    ///
    /// This method will return an [`Err`](ShapeError) if the shape is
    /// invalid (see the Errors section on [`TensorBase::zeros()`]) or if the
    /// number of elements in `data` does not match the number of elements
    /// required by the `shape`. (That is, the number of elements equals to the
    /// product of the dimensions in `shape`.)
    #[inline]
    pub fn from_vec(
        data: Vec<B::Primitive>,
        shape: &[usize],
    ) -> Result<Self, ShapeError> {
        let expected_elements = Self::get_validated_num_elements(shape)?;

        if data.len() != expected_elements {
            return Err(ShapeError::ElementCountMismatch);
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
    /// See the error notes on [`TensorBase::zeros()`].
    #[inline]
    pub fn ones(shape: &[usize]) -> Result<Self, ShapeError> {
        let _: usize = Self::get_validated_num_elements(shape)?;

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
    /// This method returns [`Err`](ShapeError) if:
    ///
    /// - one of the dimensions in the parameter value is 0,
    /// - any of the axis lengths overflows [`isize::MAX`],
    /// - the product of axis lengths overflows `isize::MAX`.
    #[inline]
    pub fn zeros(shape: &[usize]) -> Result<Self, ShapeError> {
        let _: usize = Self::get_validated_num_elements(shape)?;

        // SAFETY: `shape` does not contain any zeros, no dimensions overflow
        // `isize`, the product of dimensions does not overflow `isize::MAX`.
        let inner = unsafe { B::zeros(shape) };

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Ensures that all preconditions for creating a tensor from a shape are
    /// met, and, on success, returns the product of the dimensions (the number
    /// of elements).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - zero-sized dimensions,
    /// - any of the dimensions overflowing `isize`,
    /// - overflow when calculating the total number of elements.
    fn get_validated_num_elements(
        shape: &[usize],
    ) -> Result<usize, ShapeError> {
        if shape.contains(&0) {
            return Err(ShapeError::ZeroDim);
        }

        let num_elements = shape
            .iter()
            .try_fold(1_usize, |prod, &dim| prod.checked_mul(dim))
            .ok_or(ShapeError::ShapeOverflow)?;

        let _: isize = num_elements
            .try_into()
            .map_err(|_| ShapeError::ShapeOverflow)?;

        Ok(num_elements)
    }

    fn checked_binary_op<F>(
        &self,
        rhs: &Self,
        op: F,
    ) -> Result<Self, IncompatibleTensorsError>
    where
        F: FnOnce(&B::Tensor, &B::Tensor) -> B::Tensor,
    {
        if self.shape() != rhs.shape() {
            return Err(IncompatibleTensorsError::ShapeMismatch);
        }

        let inner = op(&self.inner, &rhs.inner);

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $checked_method:ident) => {
        impl<B: Backend> $trait<TensorBase<B>> for TensorBase<B> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                self.$checked_method(&rhs)
                    .expect("incompatible tensor shapes for operation")
            }
        }

        impl<'rhs, B: Backend> $trait<&'rhs TensorBase<B>> for TensorBase<B> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: &'rhs TensorBase<B>) -> Self::Output {
                self.$checked_method(rhs)
                    .expect("incompatible tensor shapes for operation")
            }
        }

        impl<B: Backend> $trait<TensorBase<B>> for &TensorBase<B> {
            type Output = TensorBase<B>;

            #[inline]
            fn $method(self, rhs: TensorBase<B>) -> Self::Output {
                self.$checked_method(&rhs)
                    .expect("incompatible tensor shapes for operation")
            }
        }

        impl<'rhs, B: Backend> $trait<&'rhs TensorBase<B>> for &TensorBase<B> {
            type Output = TensorBase<B>;

            #[inline]
            fn $method(self, rhs: &'rhs TensorBase<B>) -> Self::Output {
                self.$checked_method(rhs)
                    .expect("incompatible tensor shapes for operation")
            }
        }
    };
}

macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $backend_method:ident, $($t:ty),*) => {
        $(
            impl<B: Backend<Primitive = $t>> $trait<$t> for TensorBase<B> {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: $t) -> Self::Output {
                    let inner = B::$backend_method(&self.inner, rhs);
                    Self { inner, _marker: PhantomData }
                }
            }

            impl<B: Backend<Primitive = $t>> $trait<$t> for &TensorBase<B> {
                type Output = TensorBase<B>;

                #[inline]
                fn $method(self, rhs: $t) -> Self::Output {
                    let inner = B::$backend_method(&self.inner, rhs);
                    Self::Output { inner, _marker: PhantomData }
                }
            }
        )*
    };
}

impl_binary_op!(Add, add, checked_add);

impl_scalar_op!(Add, add, add_scalar, f32, f64, i8, i16, i32, i64, i128);

#[cfg(test)]
mod tests {
    use crate::tensor::{
        Backend, IncompatibleTensorsError, ShapeError, TensorBase,
    };

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

        fn add_scalar(
            tensor: &Self::Tensor,
            scalar: Self::Primitive,
        ) -> Self::Tensor {
            Self::Tensor {
                shape: tensor.shape.clone(),
                value: tensor.value + scalar,
            }
        }

        unsafe fn add(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
            Self::Tensor {
                shape: lhs.shape.clone(),
                value: lhs.value + rhs.value,
            }
        }

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

        unsafe fn mul(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
            Self::Tensor {
                shape: lhs.shape.to_owned(),
                value: lhs.value + rhs.value,
            }
        }

        fn mul_scalar(
            tensor: &Self::Tensor,
            scalar: Self::Primitive,
        ) -> Self::Tensor {
            Self::Tensor {
                shape: tensor.shape.to_owned(),
                value: tensor.value + scalar,
            }
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

        unsafe fn sub(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
            Self::Tensor {
                shape: lhs.shape.clone(),
                value: lhs.value - rhs.value,
            }
        }

        unsafe fn zeros(shape: &[usize]) -> Self::Tensor {
            Self::Tensor {
                shape: shape.to_owned(),
                value: 0.0,
            }
        }
    }

    #[test]
    fn tensor_creation_zeros_works() {
        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]);

        assert!(tensor.is_ok());
    }

    #[test]
    fn tensor_creation_zeros_has_right_shape() {
        let shape = &[2, 3];
        let tensor = TensorBase::<MockBackend>::zeros(shape).unwrap();

        assert_eq!(tensor.inner.shape, shape);
    }

    #[test]
    fn tensor_creation_zeros_has_right_value() {
        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();

        assert_eq!(tensor.inner.value, 0.0);
    }

    #[test]
    fn tensor_creation_zeros_fails_on_zero_dimension() {
        let tensor = TensorBase::<MockBackend>::zeros(&[2, 0]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ZeroDim);
    }

    #[test]
    fn tensor_creation_zeros_fails_on_overflow() {
        let isize_max = usize::try_from(isize::MAX).unwrap();
        let tensor = TensorBase::<MockBackend>::zeros(&[isize_max, 2]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ShapeOverflow);
    }

    #[test]
    fn tensor_creation_ones_works() {
        let tensor = TensorBase::<MockBackend>::ones(&[2, 3]);

        assert!(tensor.is_ok());
    }

    #[test]
    fn tensor_creation_ones_has_right_shape() {
        let shape = &[2, 3];
        let tensor = TensorBase::<MockBackend>::ones(shape).unwrap();

        assert_eq!(tensor.inner.shape, shape);
    }

    #[test]
    fn tensor_creation_ones_has_right_values() {
        let tensor = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();

        assert_eq!(tensor.inner.value, 1.0);
    }

    #[test]
    fn tensor_creation_ones_fails_on_zero_dimension() {
        let tensor = TensorBase::<MockBackend>::ones(&[2, 0]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ZeroDim);
    }

    #[test]
    fn tensor_creation_ones_fails_on_overflow() {
        let isize_max = usize::try_from(isize::MAX).unwrap();
        let tensor = TensorBase::<MockBackend>::ones(&[isize_max, 2]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ShapeOverflow);
    }

    #[test]
    fn tensor_shape_is_correct() {
        let shape = &[2, 3];
        let tensor = TensorBase::<MockBackend>::zeros(shape).unwrap();

        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn tensor_ndim_is_correct() {
        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();

        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn tensor_validation_fails_on_too_large_usize() {
        let usize_max = usize::MAX;
        let result = TensorBase::<MockBackend>::get_validated_num_elements(&[
            usize_max, 1,
        ]);

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(err, ShapeError::ShapeOverflow);
    }

    #[test]
    fn tensor_validation_fails_on_overflow() {
        let isize_max = usize::try_from(isize::MAX).unwrap();
        let result = TensorBase::<MockBackend>::get_validated_num_elements(&[
            isize_max, 3,
        ]);

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(err, ShapeError::ShapeOverflow);
    }

    #[test]
    fn tensor_validation_succeeds_on_correct_shape() {
        let result =
            TensorBase::<MockBackend>::get_validated_num_elements(&[2, 3]);

        assert!(result.is_ok());
    }

    #[test]
    fn tensor_validation_fails_on_zero_dimensions() {
        let result =
            TensorBase::<MockBackend>::get_validated_num_elements(&[2, 0]);

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(err, ShapeError::ZeroDim);
    }

    #[test]
    fn tensor_validation_returns_num_of_elements() {
        let shape = &[2, 3];
        let expected_elements: usize = shape.iter().product();

        let result =
            TensorBase::<MockBackend>::get_validated_num_elements(shape);

        assert!(result.is_ok());

        let result = result.unwrap();

        assert_eq!(result, expected_elements);
    }

    #[test]
    fn tensor_creation_from_vec_succeeds() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = &[2, 3];
        let tensor = TensorBase::<MockBackend>::from_vec(data, shape);

        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();

        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn tensor_creation_from_vec_fails_on_too_few_elements() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = TensorBase::<MockBackend>::from_vec(data, &[2, 3]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ElementCountMismatch);
    }

    #[test]
    fn tensor_creation_from_vec_fails_on_too_many_elements() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = TensorBase::<MockBackend>::from_vec(data, &[1, 2]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ElementCountMismatch);
    }

    #[test]
    fn tensor_creation_from_vec_fails_on_invalid_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorBase::<MockBackend>::from_vec(data, &[2, 0]);

        assert!(tensor.is_err());

        let tensor_err = tensor.unwrap_err();

        assert_eq!(tensor_err, ShapeError::ZeroDim);
    }

    #[test]
    fn add_returns_correct_shape() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let result = a.checked_add(&b);

        assert!(result.is_ok());
        let sum_tensor = result.unwrap();
        assert_eq!(sum_tensor.shape(), &[2, 3]);
        assert_eq!(sum_tensor.inner.value, 1.0);
    }

    #[test]
    fn add_returns_correct_values() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let result = a.checked_add(&b);

        assert!(result.is_ok());
        let sum_tensor = result.unwrap();
        assert_eq!(sum_tensor.inner.value, 1.0);
    }

    #[test]
    fn add_fails_with_mismatched_shapes() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();
        let result = a.checked_add(&b);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            IncompatibleTensorsError::ShapeMismatch
        );
    }

    #[test]
    fn owned_add_operator_succeeds_with_matching_shapes() {
        let lhs = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let rhs = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let result = lhs + rhs;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.inner.value, 1.0);
    }

    #[test]
    #[should_panic(expected = "incompatible tensor shapes for operation")]
    fn owned_add_operator_panics_with_mismatched_shapes() {
        let lhs = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let rhs = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();

        let _result = lhs + rhs;
    }

    #[test]
    fn borrowed_add_operator_succeeds_with_matching_shapes() {
        let lhs = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let rhs = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let result = &lhs + &rhs;

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.inner.value, 1.0);
    }

    #[test]
    #[should_panic(expected = "incompatible tensor shapes for operation")]
    fn borrowed_add_operator_panics_with_mismatched_shapes() {
        let lhs = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let rhs = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();

        let _result = &lhs + &rhs;
    }

    #[test]
    fn add_op_owned_ref_succeeds() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 2]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[2, 2]).unwrap();
        let c = a + &b;

        assert_eq!(c.inner.value, 1.0);
    }

    #[test]
    fn add_op_ref_owned_succeeds() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 2]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[2, 2]).unwrap();
        let c = &a + b;

        assert_eq!(c.inner.value, 1.0);
    }

    #[test]
    #[should_panic(expected = "incompatible tensor shapes for operation")]
    fn add_op_owned_ref_panics_on_mismatch() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();

        let _result = a + &b;
    }

    #[test]
    #[should_panic(expected = "incompatible tensor shapes for operation")]
    fn add_op_ref_owned_panics_on_mismatch() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();

        let _result = &a + b;
    }

    #[test]
    fn add_scalar_op_owned_succeeds() {
        let tensor = TensorBase::<MockBackend>::ones(&[2, 2]).unwrap();
        let result = tensor + 10.0;
        assert_eq!(result.inner.value, 11.0);
    }

    #[test]
    fn add_scalar_op_borrowed_succeeds() {
        let tensor = TensorBase::<MockBackend>::ones(&[2, 2]).unwrap();
        let result = &tensor + 10.0;
        assert_eq!(result.inner.value, 11.0);

        assert_eq!(tensor.inner.value, 1.0);
    }

    #[test]
    fn sub_returns_correct_shape() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let result = a.checked_sub(&b);

        assert!(result.is_ok());
        let sum_tensor = result.unwrap();
        assert_eq!(sum_tensor.shape(), &[2, 3]);
    }

    #[test]
    fn sub_returns_correct_values() {
        let a = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[2, 3]).unwrap();
        let result = a.checked_sub(&b);

        assert!(result.is_ok());
        let sum_tensor = result.unwrap();
        assert_eq!(sum_tensor.inner.value, 0.0);
    }

    #[test]
    fn sub_fails_with_mismatched_shapes() {
        let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
        let b = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();
        let result = a.checked_sub(&b);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            IncompatibleTensorsError::ShapeMismatch
        );
    }
}

#[cfg(all(test, feature = "ndarray-backend"))]
mod type_alias_tests {
    use crate::tensor::{
        FloatTensor, IncompatibleTensorsError, ShapeError, Tensor,
    };

    #[test]
    fn tensor_alias_zeros_creates_tensor_with_zeros() {
        let shape = &[2, 3];
        let tensor = Tensor::<f32>::zeros(shape).unwrap();

        assert_eq!(tensor.shape(), shape);
        assert!(tensor.inner.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn tensor_alias_ones_creates_tensor_with_ones() {
        let shape = &[2, 3];
        let tensor = Tensor::<f32>::ones(shape).unwrap();

        assert_eq!(tensor.shape(), shape);
        assert!(tensor.inner.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn tensor_alias_from_vec_creates_correct_tensor() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_clone = data.clone();
        let tensor = Tensor::<f32>::from_vec(data, shape).unwrap();

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.inner.as_slice().unwrap(), &data_clone);
    }

    #[test]
    fn tensor_alias_from_vec_fails_on_element_mismatch() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = Tensor::<f32>::from_vec(data, shape);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ShapeError::ElementCountMismatch);
    }

    #[test]
    fn tensor_alias_ndim_is_correct() {
        let shape = &[2, 3, 4];
        let tensor = Tensor::<f32>::zeros(shape).unwrap();

        assert_eq!(tensor.ndim(), 3);
    }

    #[test]
    fn tensor_alias_zeros_fails_on_zero_dim() {
        let shape = &[2, 0];
        let result = Tensor::<f32>::zeros(shape);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ShapeError::ZeroDim);
    }

    #[test]
    fn tensor_alias_works_with_other_primitives() {
        let shape = &[4, 1];
        let tensor = Tensor::<i32>::zeros(shape).unwrap();

        assert_eq!(tensor.shape(), shape);
        assert!(tensor.inner.iter().all(|&value| value == 0));
    }

    #[test]
    fn float_tensor_alias_is_tensor_f32() {
        let shape = &[2, 2];
        let tensor_f32 = Tensor::<f32>::ones(shape).unwrap();
        let float_tensor = FloatTensor::ones(shape).unwrap();

        assert_eq!(tensor_f32.shape(), float_tensor.shape());
        assert_eq!(tensor_f32.inner, float_tensor.inner);
    }

    #[test]
    fn alias_add_returns_correct_shape() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::ones(&[2, 3]).unwrap();
        let result = a.checked_add(&b);

        assert!(result.is_ok());
        let sum_tensor = result.unwrap();
        assert_eq!(sum_tensor.shape(), &[2, 3]);
    }

    #[test]
    fn alias_add_fails_with_mismatched_shapes() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::ones(&[2, 4]).unwrap();
        let result = a.checked_add(&b);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            IncompatibleTensorsError::ShapeMismatch
        );
    }
}
