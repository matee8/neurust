//! The generic computational backend.
//!
//! This module provides the [`Backend`] trait which defines the complete
//! contract for all linear algebra operations, data management, and tensor
//! creation.
//!
//! The default backend is [`ndarray`] and can be swapped out using crate
//! feature flags.

use core::ops::{Add, Div, Mul, Sub};

use num_traits::{One, Zero};

pub mod ndarray;

/// A trait that defines the contract for tensor operations that every
/// backend must fulfill.
///
/// The `Backend` trait is the core abstraction of this module. It provides a
/// generic interface for tensor creation, manipulation, and computation. By
/// implementing this trait, different computation libraries (like [`ndarray`]
/// or `nalgebra`) can be used as the underlying engine for tensor operations.
/// This allows for flexibility and performance tuning by switching backends
/// through feature flags. All functions are pure.
///
/// Some methods in this trait are marked `unsafe` because they do not perform
/// any invariant checks (e.g., for shape compatibility). The caller (typically
/// the [`TensorBase`](crate::tensor::TensorBase) wrapper) is responsible for
/// ensuring all preconditions are met before calling these functions.
pub trait Backend {
    /// The element type of the tensors.
    type Primitive: Clone
        + Copy
        + Zero
        + One
        + Add<Output = Self::Primitive>
        + Sub<Output = Self::Primitive>
        + Mul<Output = Self::Primitive>
        + Div<Output = Self::Primitive>;
    /// The concrete tensor representation provided by the backend.
    type Tensor: Clone;

    /// Adds two tensors element-wise.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `lhs` and `rhs` have the same shape.
    unsafe fn add(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor;

    /// Adds a scalar to every element of a tensor.
    fn add_scalar(
        tensor: &Self::Tensor,
        scalar: Self::Primitive,
    ) -> Self::Tensor;

    /// Divides the first tensor by the second, element-wise.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `lhs` and `rhs` have the same shape.
    unsafe fn div(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor;

    /// Divides every element of the tensor by a scalar.
    fn div_scalar(
        tensor: &Self::Tensor,
        scalar: Self::Primitive,
    ) -> Self::Tensor;

    /// Creates a tensor from a vector of data and a shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// -   The `shape` is valid (no zero dimensions, no overflow, see safety
    ///     notes on [`Backend::zeros()`]).
    /// -   The number of elements in `data` is equal to the product of the
    ///     `shape`'s dimensions.
    unsafe fn from_vec(
        data: Vec<Self::Primitive>,
        shape: &[usize],
    ) -> Self::Tensor;

    /// Performs matrix multiplication of two 2D tensors.
    ///
    /// # Safety
    ///
    /// The caller must ensure that both `lhs` and `rhs` are 2D tensors and that
    /// the inner dimensions are compatible
    /// (`lhs.shape()[1] == rhs.shape()[0]`).
    unsafe fn matmul(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor;

    /// Calculates the mean of a tensors's elements along a specified axis,
    /// removing the dimension.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `axis` is a valid dimension index for
    /// the tensor (i.e., `axis < tensor.ndim()`).
    unsafe fn mean(tensor: &Self::Tensor, axis: usize) -> Self::Tensor;

    /// Calculates the mean of a tensor's elements along a specified axis,
    /// keeping the dimension with size 1.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `axis` is a valid dimension index for
    /// the tensor (i.e., `axis < tensor.ndim()`).
    unsafe fn mean_keep_dims(
        tensor: &Self::Tensor,
        axis: usize,
    ) -> Self::Tensor;

    /// Multiplies two tensors element-wise.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `lhs` and `rhs` have the same shape.
    unsafe fn mul(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor;

    /// Multiplies every element of a tensor by a scalar.
    fn mul_scalar(
        tensor: &Self::Tensor,
        scalar: Self::Primitive,
    ) -> Self::Tensor;

    /// Returns the number of dimensions of the tensor.
    fn ndim(tensor: &Self::Tensor) -> usize;

    /// Creates a tensor with all elements set to one, with the given shape.
    ///
    /// # Safety
    ///
    /// See the safety notes for [`Backend::zeros()`].
    unsafe fn ones(shape: &[usize]) -> Self::Tensor;

    /// Changes the shape of the tensor without changing its data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the new shape is valid and has the same
    /// number of elements as the original tensor.
    unsafe fn reshape(tensor: &Self::Tensor, shape: &[usize]) -> Self::Tensor;

    /// Returns the shape of the tensor as a slice of dimensions.
    fn shape(tensor: &Self::Tensor) -> &[usize];

    /// Subtracts the second tensor from the first, element-wise.
    ///
    /// # Safety
    ///
    /// See the `Safety` notes on [`Backend::add()`].
    unsafe fn sub(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor;

    /// Substracts a scalar from every element of a tensor.
    fn sub_scalar(
        tensor: &Self::Tensor,
        scalar: Self::Primitive,
    ) -> Self::Tensor;

    /// Sums the elements of a tensor along a specific axis, removing the
    /// dimension.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `axis` is a valid dimension index for
    /// the tensor (i.e., `axis < tensor.ndim()`).
    unsafe fn sum(tensor: &Self::Tensor, axis: usize) -> Self::Tensor;

    /// Sums the elements of a tensor along a specified axis, keeping the
    /// dimension with size 1.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `axis` is a valid dimension index for
    /// the tensor (i.e., `axis < tensor.ndim()`).
    unsafe fn sum_keep_dims(tensor: &Self::Tensor, axis: usize)
    -> Self::Tensor;

    /// Transposes a 2D tensor, swapping its axes.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the tensor is 2-dimensional.
    unsafe fn transpose(tensor: &Self::Tensor) -> Self::Tensor;

    /// Creates a tensor with all elements set to zero, with the given shape.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the `shape` slice does not contain any
    /// zeros, no dimensions overflow `isize`, and the product of axis lengths
    /// does not overflow [`isize::MAX`].
    unsafe fn zeros(shape: &[usize]) -> Self::Tensor;
}

cfg_if::cfg_if! {
    if #[cfg(feature = "ndarray-backend")] {
        /// Dynamically configured type alias for the selected backend, based
        /// on crate feature flags.
        pub type SelectedBackend<T> = ndarray::NdarrayBackend<T>;
    } else {
        compile_error!(
            "A backend feature must be enabled. Available: `backend-ndarray`"
        );
    }
}
