//! [`ndarray`] crate backend.

use core::marker::PhantomData;

use ndarray::{ArrayD, IxDyn, ScalarOperand};
use num_traits::{One, Zero};

use crate::backend::Backend;

/// Marker type for the [`ndarray`] backend.
#[derive(Debug)]
pub struct NdarrayBackend<T>
where
    T: Clone,
{
    _marker: PhantomData<T>,
}

impl<T> Backend for NdarrayBackend<T>
where
    T: Clone + Zero + One + ScalarOperand,
{
    type Primitive = T;
    type Tensor = ArrayD<T>;

    #[inline]
    unsafe fn add(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
        lhs + rhs
    }

    #[inline]
    fn add_scalar(
        tensor: &Self::Tensor,
        scalar: Self::Primitive,
    ) -> Self::Tensor {
        tensor + scalar
    }

    #[inline]
    unsafe fn from_vec(
        data: Vec<Self::Primitive>,
        shape: &[usize],
    ) -> Self::Tensor {
        // SAFETY: The caller has already guaranteed that the shape is valid
        // and the element count in `data` matches the shape's requirements.
        unsafe { ArrayD::from_shape_vec_unchecked(IxDyn(shape), data) }
    }

    #[inline]
    fn ndim(tensor: &Self::Tensor) -> usize {
        tensor.ndim()
    }

    #[inline]
    unsafe fn ones(shape: &[usize]) -> Self::Tensor {
        ArrayD::ones(IxDyn(shape))
    }

    #[inline]
    fn shape(tensor: &Self::Tensor) -> &[usize] {
        tensor.shape()
    }

    #[inline]
    unsafe fn zeros(shape: &[usize]) -> Self::Tensor {
        ArrayD::zeros(IxDyn(shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, ndarray::NdarrayBackend};

    #[test]
    fn ndarray_zeros_has_correct_shape() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };

        assert_eq!(array.shape(), shape);
    }

    #[test]
    fn ndarray_zeros_has_correct_values() {
        let array = unsafe { NdarrayBackend::<f32>::zeros(&[2, 3]) };

        assert!(array.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn ndarray_ones_has_correct_shape() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::ones(shape) };

        assert_eq!(array.shape(), shape);
    }

    #[test]
    fn ndarray_ones_has_correct_values() {
        let array = unsafe { NdarrayBackend::<f32>::ones(&[2, 3]) };

        assert!(array.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn ndarray_shape_is_correct() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };

        assert_eq!(NdarrayBackend::shape(&array), shape);
    }

    #[test]
    fn ndarray_ndim_is_correct() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };

        assert_eq!(NdarrayBackend::ndim(&array), shape.len());
    }

    #[test]
    fn ndarray_from_vec_is_correct() {
        let shape = &[2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_clone = data.clone();
        let array = unsafe { NdarrayBackend::<f32>::from_vec(data, shape) };

        assert_eq!(array.shape(), shape);
        assert_eq!(array.into_raw_vec_and_offset().0, data_clone);
    }

    #[test]
    fn ndarray_add_produces_correct_shape() {
        let shape = &[2, 3];
        let lhs = unsafe { NdarrayBackend::<f32>::zeros(shape) };
        let rhs = unsafe { NdarrayBackend::<f32>::ones(shape) };

        let result = unsafe { NdarrayBackend::add(&lhs, &rhs) };

        assert_eq!(result.shape(), shape);
    }

    #[test]
    fn ndarray_add_produces_correct_values() {
        let shape = &[2, 2];
        let lhs_data = vec![1.0, 2.0, 3.0, 4.0];
        let rhs_data = vec![5.0, 6.0, 7.0, 8.0];
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        let lhs = unsafe { NdarrayBackend::from_vec(lhs_data, shape) };
        let rhs = unsafe { NdarrayBackend::from_vec(rhs_data, shape) };

        let result = unsafe { NdarrayBackend::add(&lhs, &rhs) };

        assert_eq!(result.into_raw_vec_and_offset().0, expected);
    }

    #[test]
    fn ndarray_add_scalar_preserves_shape() {
        let shape = &[2, 3, 4];
        let tensor = unsafe { NdarrayBackend::<f32>::zeros(shape) };
        let scalar = 5.0;

        let result = NdarrayBackend::add_scalar(&tensor, scalar);

        assert_eq!(result.shape(), shape);
    }

    #[test]
    fn ndarray_add_scalar_produces_correct_values() {
        let shape = &[2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let scalar = 10.0;
        let expected = vec![11.0, 12.0, 13.0, 14.0];

        let tensor = unsafe { NdarrayBackend::from_vec(data, shape) };
        let result = NdarrayBackend::add_scalar(&tensor, scalar);

        assert_eq!(result.into_raw_vec_and_offset().0, expected);
    }

    #[test]
    fn ndarray_add_scalar_works_for_integers() {
        let shape = &[2, 2];
        let data = vec![1, 2, 3, 4];
        let scalar = 10;
        let expected = vec![11, 12, 13, 14];

        let tensor = unsafe { NdarrayBackend::<i32>::from_vec(data, shape) };
        let result = NdarrayBackend::add_scalar(&tensor, scalar);

        assert_eq!(result.into_raw_vec_and_offset().0, expected);
    }
}
