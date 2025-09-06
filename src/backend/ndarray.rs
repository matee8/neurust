//! [`ndarray`] crate backend.

use core::{marker::PhantomData, ops::Sub};

use ndarray::{
    ArrayBase, ArrayD, Dim, IxDyn, IxDynImpl, OwnedRepr, ScalarOperand,
};
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
    for<'borrow> &'borrow ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>: Sub<
            &'borrow ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>,
            Output = ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>,
        >,
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
    unsafe fn mul(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
        lhs * rhs
    }

    #[inline]
    fn mul_scalar(
        tensor: &Self::Tensor,
        scalar: Self::Primitive,
    ) -> Self::Tensor {
        tensor * scalar
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
    unsafe fn sub(lhs: &Self::Tensor, rhs: &Self::Tensor) -> Self::Tensor {
        lhs - rhs
    }

    #[inline]
    unsafe fn zeros(shape: &[usize]) -> Self::Tensor {
        ArrayD::zeros(IxDyn(shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::{Backend, ndarray::NdarrayBackend};

    macro_rules! test_binary_op {
        ($name:ident, $op:ident, $lhs:expr, $rhs:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let shape = &[2, 2];
                let lhs = unsafe { NdarrayBackend::from_vec($lhs, shape) };
                let rhs = unsafe { NdarrayBackend::from_vec($rhs, shape) };

                let result_shape = unsafe { NdarrayBackend::$op(&lhs, &rhs) };
                assert_eq!(result_shape.shape(), shape);

                let result_values = unsafe { NdarrayBackend::$op(&lhs, &rhs) };
                assert_eq!(
                    result_values.into_raw_vec_and_offset().0,
                    $expected
                );
            }
        };
    }

    macro_rules! test_scalar_op {
        ($name:ident, $op:ident, $input:expr, $scalar:expr, $expected:expr) => {
            #[test]
            fn $name() {
                let shape = &[2, 2];
                let tensor =
                    unsafe { NdarrayBackend::from_vec($input.clone(), shape) };

                let result_shape = NdarrayBackend::$op(&tensor, $scalar);
                assert_eq!(result_shape.shape(), shape);

                let result_values = NdarrayBackend::$op(&tensor, $scalar);
                assert_eq!(
                    result_values.into_raw_vec_and_offset().0,
                    $expected
                );
            }
        };
    }

    #[test]
    fn ndarray_zeros_has_correct_shape_and_values() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };
        assert_eq!(array.shape(), shape);
        assert!(array.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn ndarray_ones_has_correct_shape_and_values() {
        let shape = &[2, 3];
        let array = unsafe { NdarrayBackend::<f32>::ones(shape) };
        assert_eq!(array.shape(), shape);
        assert!(array.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn ndarray_shape_and_ndim_are_correct() {
        let shape = &[2, 3, 4];
        let array = unsafe { NdarrayBackend::<f32>::zeros(shape) };
        assert_eq!(NdarrayBackend::shape(&array), shape);
        assert_eq!(NdarrayBackend::ndim(&array), 3);
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

    test_binary_op!(
        ndarray_add_produces_correct_values,
        add,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![6.0, 8.0, 10.0, 12.0]
    );

    test_binary_op!(
        ndarray_sub_produces_correct_values,
        sub,
        vec![10.0, 8.0, 6.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![9.0, 6.0, 3.0, 0.0]
    );

    test_binary_op!(
        ndarray_mul_produces_correct_values,
        mul,
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![5.0, 12.0, 21.0, 32.0]
    );

    test_scalar_op!(
        ndarray_add_scalar_produces_correct_values,
        add_scalar,
        vec![1.0, 2.0, 3.0, 4.0],
        10.0,
        vec![11.0, 12.0, 13.0, 14.0]
    );

    test_scalar_op!(
        ndarray_add_scalar_works_for_integers,
        add_scalar,
        vec![1, 2, 3, 4],
        10,
        vec![11, 12, 13, 14]
    );

    test_scalar_op!(
        ndarray_mul_scalar_produces_correct_values,
        mul_scalar,
        vec![1.0, 2.0, 3.0, 4.0],
        10.0,
        vec![10.0, 20.0, 30.0, 40.0]
    );
}
