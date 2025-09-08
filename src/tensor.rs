//! # Generic tensor module.
//!
//! This module provides a generic [`TensorBase`] structure that is
//! parameterized by a [backend](crate::backend::Backend), and convenient type
//! aliases for the most common use cases.

use core::{
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

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
    /// The tensor does not have the required number of dimensions for the
    /// operation.
    #[error("invalid tensor dimension for operation")]
    InvalidDimension,
    /// The provided axis is out of bounds for the tensor's shape.
    #[error("invalid axis for tensor with less dimensions")]
    InvalidAxis,
}

/// Generic, backend-agnostic n-dimensional tensor.
///
/// `TensorBase` is a thin wrapper around a backend-specific tensor
/// implementation. It only performs invariants checks at runtime and delegates
/// the actual maths to the backend.
///
/// For most applications, the [`Tensor`] type alias is more convenient.
#[derive(Debug, Clone)]
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
        self.checked_binary_op(other, B::add)
    }

    /// Performs element-wise division between two tensors.
    ///
    /// # Errors
    ///
    /// Returns a [`IncompatibleTensorsError::ShapeMismatch`] if the tensors
    /// do not have the same shape.
    #[inline]
    pub fn checked_div(
        &self,
        other: &Self,
    ) -> Result<Self, IncompatibleTensorsError> {
        self.checked_binary_op(other, B::div)
    }

    /// Performs matrix multiplication of two 2D tensors.
    ///
    /// # Errors
    ///
    /// Returns an [`IncompatibleTensorsError`] if:
    /// - Either `self` or `other` is not a 2-dimensional tensor.
    /// - The inner dimensions are not compatible (i.e.,
    ///   `self.shape()[1] != other.shape()[0]`).
    #[expect(
        clippy::indexing_slicing,
        reason = "It is safe to index `shape` since `ndim` is > 2."
    )]
    #[inline]
    pub fn checked_matmul(
        &self,
        other: &Self,
    ) -> Result<Self, IncompatibleTensorsError> {
        if self.ndim() != 2 {
            return Err(IncompatibleTensorsError::InvalidDimension);
        }

        if other.ndim() != 2 {
            return Err(IncompatibleTensorsError::InvalidDimension);
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(IncompatibleTensorsError::ShapeMismatch);
        }

        // SAFETY: We have verified that both tensors are 2D and their inner
        // dimensions are compatible.
        let inner = unsafe { B::matmul(&self.inner, &other.inner) };

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Performs element-wise multiplication between two tensors.
    ///
    /// # Errors
    ///
    /// Returns a [`IncompatibleTensorsError::ShapeMismatch`] if the tensors do
    /// not have the samep shape.
    #[inline]
    pub fn checked_mul(
        &self,
        other: &Self,
    ) -> Result<Self, IncompatibleTensorsError> {
        self.checked_binary_op(other, B::mul)
    }

    /// Returns a new tensor with the specified shape, without changing the
    /// underlying data.
    ///
    /// # Errors
    ///
    /// Returns a [`ShapeError`] if:
    /// - The new shape is invalid (contains a zero, overflows `isize`).
    /// - The total number of elements in the new shape does not match the
    ///   number of elements in the current tensor.
    #[inline]
    pub fn checked_reshape(&self, shape: &[usize]) -> Result<Self, ShapeError> {
        let new_num_elements = Self::get_validated_num_elements(shape)?;
        let current_num_elements = self.shape().iter().product();

        if new_num_elements != current_num_elements {
            return Err(ShapeError::ElementCountMismatch);
        }

        // SAFETY: The new shape is valid and the element count is guaranteed
        // to match the original tensor's element count.
        let inner = unsafe { B::reshape(&self.inner, shape) };

        Ok(Self {
            inner,
            _marker: PhantomData,
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
        self.checked_binary_op(other, B::sub)
    }

    /// Transpose a 2D tensor, swapping its axes.
    ///
    /// # Errors
    ///
    /// Returns an [`IncompatibleTensorsError::InvalidDimension`] if the tensor
    /// is not 2-dimensional.
    #[inline]
    pub fn checked_transpose(&self) -> Result<Self, IncompatibleTensorsError> {
        if self.ndim() != 2 {
            return Err(IncompatibleTensorsError::InvalidDimension);
        }

        // SAFETY: We have verified that the tensor is 2D.
        let inner = unsafe { B::transpose(&self.inner) };

        Ok(Self {
            inner,
            _marker: PhantomData,
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

    /// Performs matrix multiplication of two 2D tensors.
    ///
    /// # Panics
    ///
    /// Panics if the tensor shapes are not compatible for matrix
    /// multiplication. See [`Self::checked_matmul`] for details.
    #[inline]
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = r#"The panic is documented, and end users could use the checked
                    version instead, `checked_matmul`."#
    )]
    pub fn matmul(&self, other: &Self) -> Self {
        self.checked_matmul(other)
            .expect("incompatible tensor shapes for matrix multiplication")
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

    /// Returns a new tensor with the specified shape, without changing the
    /// underlying data.
    ///
    /// # Panics
    ///
    /// Panics if the new shape is not compatible with the total number of
    /// elements in the tensor. See [`Self::checked_reshape()`] for more
    /// details.
    #[inline]
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = r#"The panic is documented, and end users could use the checked
                    version instead, `checked_reshape`."#
    )]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        self.checked_reshape(shape)
            .expect("incompatible shape for reshape operation.")
    }

    /// Returns the shape of the tensor as a slice of dimensions.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        B::shape(&self.inner)
    }

    /// Transpose a 2D tensor, swapping its axes.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2-dimensional.
    #[inline]
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = r#"The panic is documented, and end users could use the checked
                    version instead, `checked_transpose`."#
    )]
    pub fn transpose(&self) -> Self {
        self.checked_transpose()
            .expect("transpose is only valid for 2D tensors")
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

    /// A generic helper function for operations that reduce a tensor along a
    /// single axis.
    ///
    /// # Errors
    ///
    /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if `axis` is out
    /// of bounds for the tensor (i.e., `axis >= self.ndim()`).
    ///
    /// # Safety
    ///
    /// The provided `op` function is only called after `axis` has been
    /// validated, fulfilling the safety contract of the backend's axis-based
    /// methods.
    fn checked_axis_op(
        &self,
        axis: usize,
        op: unsafe fn(&B::Tensor, usize) -> B::Tensor,
    ) -> Result<Self, IncompatibleTensorsError> {
        if axis >= self.ndim() {
            return Err(IncompatibleTensorsError::InvalidAxis);
        }

        // SAFETY: We have verified that the axis is valid.
        let inner = unsafe { op(&self.inner, axis) };

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    fn checked_binary_op(
        &self,
        rhs: &Self,
        op: unsafe fn(&B::Tensor, &B::Tensor) -> B::Tensor,
    ) -> Result<Self, IncompatibleTensorsError> {
        if self.shape() != rhs.shape() {
            return Err(IncompatibleTensorsError::ShapeMismatch);
        }

        // SAFETY: The shapes are guaranteed to be the same.
        let inner = unsafe { op(&self.inner, &rhs.inner) };

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
impl_binary_op!(Sub, sub, checked_sub);
impl_binary_op!(Mul, mul, checked_mul);
impl_binary_op!(Div, div, checked_div);

impl_scalar_op!(Add, add, add_scalar, f32, f64, i8, i16, i32, i64, i128);
impl_scalar_op!(Sub, sub, sub_scalar, f32, f64, i8, i16, i32, i64, i128);
impl_scalar_op!(Mul, mul, mul_scalar, f32, f64, i8, i16, i32, i64, i128);
impl_scalar_op!(Div, div, div_scalar, f32, f64, i8, i16, i32, i64, i128);

macro_rules! impl_axis_op {
    (
        $(#[$checked_name_meta:meta])*
        $checked_name:ident,
        $backend_name:ident,

        $(#[$name_meta:meta])*
        $name:ident,

        $(#[$checked_keep_dims_name_meta:meta])*
        $checked_keep_dims_name:ident,
        $backend_keep_dims_name:ident,

        $(#[$keep_dims_name_meta:meta])*
        $keep_dims_name:ident
    ) => {
        $(#[$checked_name_meta])*
        #[inline]
        pub fn $checked_name(
            &self,
            axis: usize,
        ) -> Result<Self, IncompatibleTensorsError> {
            self.checked_axis_op(axis, B::$backend_name)
        }

        $(#[$name_meta])*
        #[inline]
        #[must_use]
        pub fn $name(&self, axis: usize) -> Self {
            self.$checked_name(axis)
                .expect("axis is out of bounds for operation")
        }

        $(#[$checked_keep_dims_name_meta])*
        #[inline]
        pub fn $checked_keep_dims_name(
            &self,
            axis: usize,
        ) -> Result<Self, IncompatibleTensorsError> {
            self.checked_axis_op(axis, B::$backend_keep_dims_name)
        }

        $(#[$keep_dims_name_meta])*
        #[inline]
        #[must_use]
        pub fn $keep_dims_name(&self, axis: usize) -> Self {
            self.$checked_keep_dims_name(axis)
                .expect("axis is out of bounds for operation")
        }
    };
}

impl<B: Backend> TensorBase<B> {
    impl_axis_op!(
        /// Sums the elements of the tensor along the specified axis, removing
        /// that dimension.
        ///
        /// # Errors
        ///
        /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if the axis is
        /// out of bounds.
        checked_sum,
        sum,
        /// Sums the elements of the tensor along the specified axis, removing
        /// that dimension
        ///
        /// # Panics
        ///
        /// Panics if the axis is out of bounds for the tensor.
        sum,
        /// Sums the elements of the tensor along the specified axis, keeping
        /// the dimension with size 1.
        ///
        /// # Errors
        ///
        /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if the axis is
        /// out of bounds.
        checked_sum_keep_dims,
        sum_keep_dims,
        /// Sums the elements of the tensor along a the specified axis, keeping
        /// the dimension with size 1.
        ///
        /// # Panics
        ///
        /// Panics if the axis is out of bounds for the tensor.
        sum_keep_dims
    );

    impl_axis_op!(
        /// Calculates the mean of the tensor's elements along the specified
        /// axis, removing that dimension.
        ///
        /// # Errors
        ///
        /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if the axis is
        /// out of bounds.
        checked_mean,
        mean,
        /// Calculates the mean of the tensor's elements along the specified
        /// axis, removing that dimension.
        ///
        /// # Panics
        ///
        /// Panics if the axis is out of bounds for the tensor.
        mean,
        /// Calculates the mean of the tensor's elements along the specified
        /// axis, keeping the dimension with size 1.
        ///
        /// # Errors
        ///
        /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if the axis if
        /// out of bounds.
        checked_mean_keep_dims,
        mean_keep_dims,
        /// Calculates the mean of the tensor's elements along the specified
        /// axis, keeping the dimension with size 1.
        ///
        /// # Panics
        ///
        /// Panics if the axis is out of bounds for the tensor.
        mean_keep_dims
    );

    impl_axis_op!(
        /// Finds the maximum elements of the tensor along the specified axis,
        /// removing that dimension.
        ///
        /// # Errors
        ///
        /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if the axis is
        /// out of bounds.
        checked_max,
        max,
        /// Finds the maximum elements of the tensor along the specified axis,
        /// removing that dimension.
        ///
        /// # Panics
        ///
        /// Panics if the axis is out of bounds for the tensor.
        max,
        /// Finds the maximum elements of the tensor along the specified axis,
        /// keeping the dimension with size 1.
        ///
        /// # Errors
        ///
        /// Returns an [`IncompatibleTensorsError::InvalidAxis`] if the axis is
        /// out of bounds.
        checked_max_keep_dims,
        max_keep_dims,
        /// Finds the maximum elements of the tensor along the specified axis,
        /// keeping the dimension with size 1.
        ///
        /// # Panics
        ///
        /// Panics if the axis is out of bounds for the tensor.
        max_keep_dims
    );
}

#[cfg(test)]
mod tests {
    use crate::backend::Backend;

    #[derive(Debug, Clone)]
    struct MockBackend;

    #[derive(Debug, Clone)]
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

        unsafe fn div(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
            Self::Tensor {
                shape: lhs.shape.clone(),
                value: lhs.value / rhs.value,
            }
        }

        fn div_scalar(
            tensor: &Self::Tensor,
            scalar: Self::Primitive,
        ) -> Self::Tensor {
            Self::Tensor {
                shape: tensor.shape.to_owned(),
                value: tensor.value / scalar,
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

        unsafe fn matmul(
            lhs: &Self::Tensor,
            rhs: &Self::Tensor,
        ) -> Self::Tensor {
            let new_shape = vec![lhs.shape[0], rhs.shape[1]];
            Self::Tensor {
                shape: new_shape,
                value: lhs.value * rhs.value,
            }
        }

        unsafe fn max(tensor: &Self::Tensor, axis: usize) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            let _ = new_shape.remove(axis);
            Self::Tensor {
                shape: new_shape,
                value: tensor.value,
            }
        }

        unsafe fn max_keep_dims(
            tensor: &Self::Tensor,
            axis: usize,
        ) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            new_shape[axis] = 1;
            Self::Tensor {
                shape: new_shape,
                value: tensor.value,
            }
        }

        unsafe fn mean(tensor: &Self::Tensor, axis: usize) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            let _ = new_shape.remove(axis);
            Self::Tensor {
                shape: new_shape,
                value: tensor.value,
            }
        }

        unsafe fn mean_keep_dims(
            tensor: &Self::Tensor,
            axis: usize,
        ) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            new_shape[axis] = 1;
            Self::Tensor {
                shape: new_shape,
                value: tensor.value,
            }
        }

        unsafe fn mul(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
            Self::Tensor {
                shape: lhs.shape.to_owned(),
                value: lhs.value * rhs.value,
            }
        }

        fn mul_scalar(
            tensor: &Self::Tensor,
            scalar: Self::Primitive,
        ) -> Self::Tensor {
            Self::Tensor {
                shape: tensor.shape.to_owned(),
                value: tensor.value * scalar,
            }
        }

        unsafe fn ones(shape: &[usize]) -> Self::Tensor {
            Self::Tensor {
                shape: shape.to_owned(),
                value: 1.0,
            }
        }

        unsafe fn reshape(
            tensor: &Self::Tensor,
            shape: &[usize],
        ) -> Self::Tensor {
            Self::Tensor {
                shape: shape.to_owned(),
                value: tensor.value,
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

        fn sub_scalar(
            tensor: &Self::Tensor,
            scalar: Self::Primitive,
        ) -> Self::Tensor {
            Self::Tensor {
                shape: tensor.shape.clone(),
                value: tensor.value - scalar,
            }
        }

        unsafe fn sum(tensor: &Self::Tensor, axis: usize) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            let _ = new_shape.remove(axis);
            Self::Tensor {
                shape: new_shape,
                value: tensor.value * tensor.shape[axis] as f32,
            }
        }

        unsafe fn sum_keep_dims(
            tensor: &Self::Tensor,
            axis: usize,
        ) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            new_shape[axis] = 1;
            Self::Tensor {
                shape: new_shape,
                value: tensor.value * tensor.shape[axis] as f32,
            }
        }

        unsafe fn transpose(tensor: &Self::Tensor) -> Self::Tensor {
            let mut new_shape = tensor.shape.clone();
            new_shape.reverse();
            Self::Tensor {
                shape: new_shape,
                value: tensor.value,
            }
        }

        unsafe fn zeros(shape: &[usize]) -> Self::Tensor {
            Self::Tensor {
                shape: shape.to_owned(),
                value: 0.0,
            }
        }
    }

    mod creation {
        use crate::tensor::{ShapeError, TensorBase, tests::MockBackend};

        macro_rules! test_creation_failure {
            ($test_name:ident, $invalid_shape:expr, $expected_error:expr) => {
                paste::paste! {
                    #[test]
                    fn [<creation_fails_on_ $test_name>]() {
                        let invalid_shape = $invalid_shape;
                        let err = $expected_error;

                        assert_eq!(
                            TensorBase::<MockBackend>::zeros(invalid_shape).unwrap_err(),
                            err
                        );
                        assert_eq!(
                            TensorBase::<MockBackend>::ones(invalid_shape).unwrap_err(),
                            err
                        );
                        assert_eq!(
                            TensorBase::<MockBackend>::from_vec(vec![], invalid_shape)
                                .unwrap_err(),
                            err
                        );
                    }
                }
            };
        }

        #[test]
        fn zeros_creates_tensor_with_correct_shape_and_value() {
            let shape = &[2, 3];
            let tensor = TensorBase::<MockBackend>::zeros(shape).unwrap();

            assert_eq!(tensor.inner.shape, shape);
            assert_eq!(tensor.inner.value, 0.0);
        }

        #[test]
        fn ones_creates_tensor_with_correct_shape_and_value() {
            let shape = &[2, 3];
            let tensor = TensorBase::<MockBackend>::ones(shape).unwrap();

            assert_eq!(tensor.inner.shape, shape);
            assert_eq!(tensor.inner.value, 1.0);
        }

        #[test]
        fn from_vec_succeeds_with_correct_inputs() {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let shape = &[2, 3];
            let tensor = TensorBase::<MockBackend>::from_vec(data, shape);
            assert!(tensor.is_ok());
            assert_eq!(tensor.unwrap().shape(), shape);
        }

        #[test]
        fn from_vec_fails_on_element_mismatch() {
            let data = vec![1.0, 2.0, 3.0];
            let result = TensorBase::<MockBackend>::from_vec(data, &[2, 3]);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), ShapeError::ElementCountMismatch);
        }

        #[test]
        fn ndim_is_correct() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3, 4]).unwrap();
            assert_eq!(tensor.ndim(), 3);
        }

        test_creation_failure!(zero_dim_shape, &[2, 0], ShapeError::ZeroDim);

        test_creation_failure!(
            overflowing_shape,
            &[usize::try_from(isize::MAX).unwrap(), 2],
            ShapeError::ShapeOverflow
        );
    }

    mod validation {
        use crate::tensor::{ShapeError, TensorBase, tests::MockBackend};

        #[test]
        fn succeeds_on_valid_shape() {
            let result =
                TensorBase::<MockBackend>::get_validated_num_elements(&[2, 3]);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), 6);
        }

        #[test]
        fn fails_on_zero_dim() {
            let result =
                TensorBase::<MockBackend>::get_validated_num_elements(&[2, 0]);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), ShapeError::ZeroDim);
        }

        #[test]
        fn fails_on_overflow() {
            let isize_max = usize::try_from(isize::MAX).unwrap();
            let result =
                TensorBase::<MockBackend>::get_validated_num_elements(&[
                    isize_max, 2,
                ]);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), ShapeError::ShapeOverflow);
        }

        #[test]
        fn fails_on_product_overflow() {
            let shape = &[usize::MAX / 2 + 1, 2];
            let result =
                TensorBase::<MockBackend>::get_validated_num_elements(shape);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), ShapeError::ShapeOverflow);
        }
    }

    mod ops {
        use core::ops::{Add, Div, Mul, Sub};

        use crate::tensor::{
            IncompatibleTensorsError, ShapeError, TensorBase,
            tests::MockBackend,
        };

        macro_rules! test_binary_op {
            (
                $test_name:ident,
                $checked_method:ident,
                $op_trait:ident,
                $op_method:ident,
                $a_val:expr,
                $b_val:expr,
                $expected_val:expr
            ) => {
                paste::paste! {
                    #[test]
                    fn [<$test_name _succeeds_for_all_ownership_combos>]() {
                        let a = TensorBase::<MockBackend>::from_vec(vec![$a_val; 4], &[2, 2]).unwrap();
                        let b = TensorBase::<MockBackend>::from_vec(vec![$b_val; 4], &[2, 2]).unwrap();

                        assert_eq!(a.$checked_method(&b).unwrap().inner.value, $expected_val);

                        assert_eq!((&a).$op_method(&b).inner.value, $expected_val);

                        assert_eq!((a.clone()).$op_method(&b).inner.value, $expected_val);

                        assert_eq!((&a).$op_method(b.clone()).inner.value, $expected_val);

                        assert_eq!(a.$op_method(b).inner.value, $expected_val);
                    }

                    #[test]
                    fn [<$test_name _checked_fails_on_mismatch>]() {
                        let c = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
                        let d = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();
                        let err_result = c.$checked_method(&d);
                        assert!(err_result.is_err());
                        assert_eq!(err_result.unwrap_err(), IncompatibleTensorsError::ShapeMismatch);
                    }

                    #[test]
                    #[should_panic(expected = "incompatible tensor shapes for operation")]
                    fn [<$test_name _op_panics_on_mismatch>]() {
                         let c = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
                         let d = TensorBase::<MockBackend>::ones(&[3, 2]).unwrap();
                         let _result = (&c).$op_method(&d);
                    }
                }
            };
        }

        macro_rules! test_scalar_op {
            (
                $test_name:ident,
                $op_trait:ident,
                $op_method:ident,
                $initial_val:expr,
                $scalar:expr,
                $expected_val:expr
            ) => {
                paste::paste! {
                    #[test]
                    fn [<$test_name _owned_succeeds>]() {
                        let tensor = TensorBase::<MockBackend>::from_vec(vec![$initial_val; 4], &[2, 2]).unwrap();
                        let result = tensor.$op_method($scalar);
                        assert_eq!(result.inner.value, $expected_val);
                    }

                    #[test]
                    fn [<$test_name _borrowed_succeeds>]() {
                        let tensor = TensorBase::<MockBackend>::from_vec(vec![$initial_val; 4], &[2, 2]).unwrap();
                        let result = (&tensor).$op_method($scalar);
                        assert_eq!(result.inner.value, $expected_val);
                        assert_eq!(tensor.inner.value, $initial_val);
                    }
                }
            };
        }

        macro_rules! test_axis_op {
            (
                $name:ident,
                $checked_name:ident,
                $keep_dims_name:ident,
                $checked_keep_dims_name:ident
            ) => {
                paste::paste! {
                    #[test]
                    fn [<$name _succeeds_on_valid_axis>]() {
                        let tensor =
                            TensorBase::<MockBackend>::from_vec(vec![1.0; 6], &[2, 3])
                                .unwrap();
                        let result = tensor.$checked_name(1).unwrap();
                        assert_eq!(result.shape(), &[2]);
                    }

                    #[test]
                    fn [<$keep_dims_name _succeeds_on_valid_axis>]() {
                        let tensor =
                            TensorBase::<MockBackend>::from_vec(vec![1.0; 6], &[2, 3])
                                .unwrap();
                        let result = tensor.$checked_keep_dims_name(1).unwrap();
                        assert_eq!(result.shape(), &[2, 1]);
                    }

                    #[test]
                    fn [<$name _fails_on_invalid_axis>]() {
                        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
                        let err = tensor.$checked_name(2).unwrap_err();
                        assert_eq!(err, IncompatibleTensorsError::InvalidAxis);
                    }

                    #[test]
                    fn [<$keep_dims_name _fails_on_invalid_axis>]() {
                        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
                        let err = tensor.$checked_keep_dims_name(2).unwrap_err();
                        assert_eq!(err, IncompatibleTensorsError::InvalidAxis);
                    }

                    #[test]
                    #[should_panic(expected = "axis is out of bounds for operation")]
                    fn [<$name _panics_on_invalid_axis>]() {
                        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
                        let _result = tensor.$name(2);
                    }

                    #[test]
                    #[should_panic(expected = "axis is out of bounds for operation")]
                    fn [<$keep_dims_name _panics_on_invalid_axis>]() {
                        let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
                        let _result = tensor.$keep_dims_name(2);
                    }
                }
            };
        }

        test_binary_op!(test_add, checked_add, Add, add, 2.0, 3.0, 5.0);
        test_binary_op!(test_sub, checked_sub, Sub, sub, 5.0, 3.0, 2.0);
        test_binary_op!(test_mul, checked_mul, Mul, mul, 2.0, 3.0, 6.0);
        test_binary_op!(div, checked_div, Div, div, 10.0, 2.0, 5.0);

        test_scalar_op!(add, Add, add, 1.0, 10.0, 11.0);
        test_scalar_op!(sub, Sub, sub, 10.0, 5.0, 5.0);
        test_scalar_op!(mul, Mul, mul, 3.0, 4.0, 12.0);
        test_scalar_op!(div, Div, div, 20.0, 4.0, 5.0);

        test_axis_op!(sum, checked_sum, sum_keep_dims, checked_sum_keep_dims);
        test_axis_op!(
            mean,
            checked_mean,
            mean_keep_dims,
            checked_mean_keep_dims
        );
        test_axis_op!(max, checked_max, max_keep_dims, checked_max_keep_dims);

        #[test]
        fn matmul_succeeds_on_valid_shapes() {
            let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let b = TensorBase::<MockBackend>::zeros(&[3, 4]).unwrap();
            let result = a.checked_matmul(&b);
            assert!(result.is_ok());
            let tensor = result.unwrap();
            assert_eq!(tensor.shape(), &[2, 4]);
        }

        #[test]
        fn matmul_fails_on_invalid_lhs_dimension() {
            let a = TensorBase::<MockBackend>::zeros(&[2, 3, 1]).unwrap();
            let b = TensorBase::<MockBackend>::zeros(&[3, 4]).unwrap();
            let err = a.checked_matmul(&b).unwrap_err();
            assert_eq!(err, IncompatibleTensorsError::InvalidDimension);
        }

        #[test]
        fn matmul_fails_on_invalid_rhs_dimension() {
            let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let b = TensorBase::<MockBackend>::zeros(&[3, 4, 1]).unwrap();
            let err = a.checked_matmul(&b).unwrap_err();
            assert_eq!(err, IncompatibleTensorsError::InvalidDimension);
        }

        #[test]
        fn matmul_fails_on_shape_mismatch() {
            let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let b = TensorBase::<MockBackend>::zeros(&[4, 5]).unwrap();
            let err = a.checked_matmul(&b).unwrap_err();
            assert_eq!(err, IncompatibleTensorsError::ShapeMismatch);
        }

        #[test]
        #[should_panic(
            expected = "incompatible tensor shapes for matrix multiplication"
        )]
        fn matmul_panics_on_mismatch() {
            let a = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let b = TensorBase::<MockBackend>::zeros(&[4, 5]).unwrap();
            let _result = a.matmul(&b);
        }

        #[test]
        fn reshape_succeeds_on_valid_shape() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let reshaped = tensor.checked_reshape(&[3, 2]);
            assert!(reshaped.is_ok());
            assert_eq!(reshaped.unwrap().shape(), &[3, 2]);
        }

        #[test]
        fn reshape_fails_on_mismatched_element_count() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let err = tensor.checked_reshape(&[4, 2]).unwrap_err();
            assert_eq!(err, ShapeError::ElementCountMismatch);
        }

        #[test]
        fn reshape_fails_on_invalid_target_shape() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let err = tensor.checked_reshape(&[6, 0]).unwrap_err();
            assert_eq!(err, ShapeError::ZeroDim);
        }

        #[test]
        #[should_panic(expected = "incompatible shape for reshape operation")]
        fn reshape_panics_on_mismatch() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let _result = tensor.reshape(&[1, 7]);
        }

        #[test]
        fn transpose_succeeds_on_2d_tensor() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3]).unwrap();
            let transposed = tensor.checked_transpose().unwrap();
            assert_eq!(transposed.shape(), &[3, 2]);
        }

        #[test]
        fn transpose_fails_on_non_2d_tensor() {
            let tensor = TensorBase::<MockBackend>::zeros(&[2, 3, 4]).unwrap();
            let err = tensor.checked_transpose().unwrap_err();
            assert_eq!(err, IncompatibleTensorsError::InvalidDimension);
        }

        #[test]
        #[should_panic(expected = "transpose is only valid for 2D tensors")]
        fn transpose_panics_on_non_2d_tensor() {
            let tensor = TensorBase::<MockBackend>::zeros(&[1, 2, 3]).unwrap();
            let _result = tensor.transpose();
        }
    }
}
