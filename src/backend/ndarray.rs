//! [`ndarray`] crate backend.

use core::marker::PhantomData;

use ndarray::{ArrayD, IxDyn};
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
    T: Clone + Zero + One,
{
    type Primitive = T;
    type Tensor = ArrayD<T>;

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
}
