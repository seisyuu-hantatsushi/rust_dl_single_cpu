
use cuda_ffi::cudnn;
use rand::SeedableRng;
use rand_distr::{Uniform, Distribution};
use rand_xorshift::XorShiftRng;

#[test]
fn tensor_add() -> Result<(),Box<dyn std::error::Error>> {
    let mut rng = XorShiftRng::from_entropy();
    let uniform_dist = Uniform::new(-2.0,2.0);
    let cudnn_context = cudnn::context::Context::new()?;
    let tensorASrc:Vec<f32> = (0..8*6).map(|_| uniform_dist.sample(&mut rng)).collect();
    let tensorBSrc:Vec<f32> = (0..8*6).map(|_| uniform_dist.sample(&mut rng)).collect();
    let tensorA = cudnn::tensor::Tensor::from_vector(cuda_ffi::CUDNNDataType::FLOAT,
						     &[8,6], tensorASrc.clone())?;
    println!("tensorA:{}\n", tensorA);
    let tensorB = cudnn::tensor::Tensor::from_vector(cuda_ffi::CUDNNDataType::FLOAT,
						     &[8,6], tensorBSrc.clone())?;
    println!("tensorB:{}\n", tensorB);

    let tensorC = cudnn_context.add(&tensorA, &tensorB)?;

    println!("tensorC:{}\n", tensorC);

    let tensorCDst:Vec<f32> = tensorC.getValues()?;
    for (&c, (&a,&b)) in tensorCDst.iter().zip(tensorASrc.iter().zip(tensorBSrc.iter())) {
	assert_eq!(c, a+b);
    }
    
    Ok(())
}
