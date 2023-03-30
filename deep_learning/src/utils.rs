/* -*- tab-width:4 -*- */


use linear_transform::tensor::Tensor;

pub fn accuracy<T,S>(predicts:&Tensor<T>, answers:&Tensor<T>) -> S
where T:Clone + std::cmp::PartialEq, S:num::Float+num::FromPrimitive {
	assert_eq!(predicts.shape(), answers.shape());

	let num_of_correct:S =
		predicts.buffer().iter().zip(answers.buffer().iter()).
		fold(num::zero(), |n,(a,p)| {
			n + if a == p { num::one() } else { num::zero() }
		});

	num_of_correct / num::FromPrimitive::from_usize(predicts.buffer().len()).unwrap()
}
