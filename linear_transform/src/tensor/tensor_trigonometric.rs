use num;
use crate::tensor::tensor_base::Tensor;

impl<T> Tensor<T>
where T:num::Float+Clone+Copy {

    pub fn sin(&self) -> Tensor<T> {
	let b:&[T] = self.buffer();
	let v = b.iter().map(|b| b.sin()).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }

    pub fn cos(&self) -> Tensor<T> {
	let b:&[T] = self.buffer();
	let v = b.iter().map(|b| b.cos()).collect::<Vec<T>>();
	Tensor::from_vector(self.shape().to_vec(), v)
    }

}
