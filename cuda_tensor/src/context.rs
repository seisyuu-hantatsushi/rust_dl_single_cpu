use cuda_binding::*;
use std::{env,mem};
use std::rc::Rc;

use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Context {
    dev_context: cuda::SharableContext,
    add_func: (Rc<cuda::Module>, cuda::Function),
    sub_func: (Rc<cuda::Module>, cuda::Function),
    hadamard_func: (Rc<cuda::Module>, cuda::Function),
    matmul_func: (Rc<cuda::Module>, cuda::Function),
    matmul_block_size: usize
}

impl Context {

    fn load_mutmal_function(module: Rc<cuda::Module>) -> Result<(usize, cuda::Function), CUDAError> {
	let mut blocksize:usize = 32;
	let matrix_muls = [
	    //"matrixMul_bs32_64bit",
	    //"matrixMul_bs16_64bit",
	    "matrixMul_bs8_64bit",
	];

	for matrix_mul in matrix_muls {
	    let func     = module.get_function(matrix_mul)?;
	    let (_gridsize, thread_per_block) =
		cuda::occupancy::max_potential_blocksize(&func, 2*blocksize*blocksize*mem::size_of::<f32>(), 0)?;
	    //println!("{},{}", _gridsize, thread_per_block);
	    if blocksize*blocksize <= thread_per_block as usize {
		return Ok((blocksize, func));
	    }
	    blocksize = blocksize/2;
	};

	Err(CUDAError::NotFound)
    }

    fn load_tensor_mutmal_function(module: Rc<cuda::Module>) -> Result<(usize, cuda::Function), CUDAError> {
	let mut blocksize:usize = 32;
	let matmuls = [
	    "tensor_matmul_bs32",
	    "tensor_matmul_bs16",
	    "tensor_matmul_bs8",
	];

	for matmul in matmuls {
	    let func     = module.get_function(matmul)?;
	    let (_gridsize, thread_per_block) =
		cuda::occupancy::max_potential_blocksize(&func, 2*blocksize*blocksize*mem::size_of::<f32>(), 0)?;
	    //println!("{},{}", _gridsize, thread_per_block);
	    if blocksize*blocksize <= thread_per_block as usize {
		return Ok((blocksize, func));
	    }
	    blocksize = blocksize/2;
	};

	Err(CUDAError::NotFound)
    }

    pub fn new(dev_context: cuda::SharableContext) -> Result<Context, CUDAError> {
        let fatbins = [include_bytes!(concat!(env!("OUT_DIR"), "/kernel.fatbin")).to_vec()];

        //dev_context.push()?;
        let module   = {
	    let m = cuda::Module::new(&fatbins[0])?;
	    Rc::new(m)
	};
        let func     = module.get_function("tensor_add")?;
        let add_func = (Rc::clone(&module), func);
        let func     = module.get_function("tensor_sub")?;
	let sub_func = (Rc::clone(&module), func);
        let func     = module.get_function("hadamard_product")?;
	let hadamard_func = (Rc::clone(&module), func);

	//let (blocksize, func) = Self::load_mutmal_function(Rc::clone(&module))?;
	let (blocksize, func) = Self::load_tensor_mutmal_function(Rc::clone(&module))?;
	let matmul_func = (module, func);
	//println!("matmul block size:{}", blocksize);
	//cuda::Context::pop()?;
        Ok(Context {
            dev_context,
            add_func,
	    sub_func,
	    hadamard_func,
	    matmul_func,
	    matmul_block_size: blocksize
        })
    }

    pub fn create_tensor<'a, T:num::Num>(&'a self, dims:&[usize]) -> Result<Tensor<'a, T>,CUDAError> {
	Tensor::<T>::new(self, dims)
    }

    pub fn add<'a, T:num::Num>(&'a self, x:&Tensor<'a, T>, y:&Tensor<'a, T>) -> Result<Tensor<'a, T>,CUDAError> {
	assert_eq!(x.shape(), y.shape());
	let shape = x.shape();
	let num_of_elements = shape.iter().fold(1,|p,&e| p*e);
	let mut z_tensor = Tensor::<T>::new(self, shape)?;
	{
	    let mut x_devmem = x.as_ref_devmem().borrow_mut();
	    let mut y_devmem = y.as_ref_devmem().borrow_mut();
	    let mut z_devmem = z_tensor.as_ref_devmem().borrow_mut();
	    //println!("add num_of_elements {}", num_of_elements);
	    assert!(shape.len() > 0);
	    let thread_per_block = 256;
	    let blocks_per_grid = (num_of_elements + thread_per_block - 1)/thread_per_block;
	    let args = vec![ cuda::execute::LaunchKernelArg::DeviceMemory(&x_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&y_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&z_devmem),
			     cuda::execute::LaunchKernelArg::Int(num_of_elements as i32)];
	    let result = cuda::execute::launch_kernel(&self.add_func.1,
						      (blocks_per_grid, 1, 1),
						      (thread_per_block, 1, 1),
						      args);

	    if let Err(err) = result {
		return Err(err);
	    }
	}
	Ok(z_tensor)
    }

    pub fn sub<'a, T:num::Num>(&'a self, x:&Tensor<'a, T>, y:&Tensor<'a, T>) -> Result<Tensor<'a, T>,CUDAError> {
	assert_eq!(x.shape(), y.shape());
	let shape = x.shape();
	let num_of_elements = shape.iter().fold(1,|p,&e| p*e);
	let mut z_tensor = Tensor::<T>::new(self, shape)?;
	{
	    let mut x_devmem = x.as_ref_devmem().borrow_mut();
	    let mut y_devmem = y.as_ref_devmem().borrow_mut();
	    let mut z_devmem = z_tensor.as_ref_devmem().borrow_mut();
	    //println!("add num_of_elements {}", num_of_elements);
	    assert!(shape.len() > 0);
	    let thread_per_block = 256;
	    let blocks_per_grid = (num_of_elements + thread_per_block - 1)/thread_per_block;
	    let args = vec![ cuda::execute::LaunchKernelArg::DeviceMemory(&x_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&y_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&z_devmem),
			     cuda::execute::LaunchKernelArg::Int(num_of_elements as i32)];
	    let result = cuda::execute::launch_kernel(&self.sub_func.1,
						      (blocks_per_grid, 1, 1),
						      (thread_per_block, 1, 1),
						      args);

	    if let Err(err) = result {
		return Err(err);
	    }
	}
	Ok(z_tensor)
    }

    pub fn hadamard_product<'a, T:num::Num>(&'a self, x:&Tensor<'a, T>, y:&Tensor<'a, T>) ->
	Result<Tensor<'a, T>,CUDAError> {
	assert_eq!(x.shape(), y.shape());
	let shape = x.shape();
	let num_of_elements = shape.iter().fold(1,|p,&e| p*e);
	let z_tensor = Tensor::<T>::new(self, shape)?;
	{
	    let x_devmem = x.as_ref_devmem().borrow();
	    let y_devmem = y.as_ref_devmem().borrow();
	    let z_devmem = z_tensor.as_ref_devmem().borrow();
	    //println!("add num_of_elements {}", num_of_elements);
	    assert!(shape.len() > 0);
	    let thread_per_block = 256;
	    let blocks_per_grid = (num_of_elements + thread_per_block - 1)/thread_per_block;
	    let args = vec![ cuda::execute::LaunchKernelArg::DeviceMemory(&x_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&y_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&z_devmem),
			     cuda::execute::LaunchKernelArg::Int(num_of_elements as i32)];
	    let result = cuda::execute::launch_kernel(&self.hadamard_func.1,
						      (blocks_per_grid, 1, 1),
						      (thread_per_block, 1, 1),
						      args);

	    if let Err(err) = result {
		return Err(err);
	    }
	}
	Ok(z_tensor)
    }

    pub fn matmul<'a, T:num::Num>(&'a self, x:&Tensor<'a, T>, y:&Tensor<'a, T>) ->
	Result<Tensor<'a, T>, CUDAError>
    {
	assert_eq!(x.shape().len(), 2);
	assert_eq!(y.shape().len(), 2);
	assert_eq!(x.shape()[1], y.shape()[0]);
	let dst_shape = vec![x.shape()[0],y.shape()[1]];
	let z_tensor = Tensor::<T>::new(self, &dst_shape)?;
	{
	    let block = (self.matmul_block_size, self.matmul_block_size, 1);
	    let grid  = (y.shape()[1]/self.matmul_block_size+1, x.shape()[0]/self.matmul_block_size+1, 1);
	    let x_devmem = x.as_ref_devmem().borrow();
	    let y_devmem = y.as_ref_devmem().borrow();
	    let z_devmem = z_tensor.as_ref_devmem().borrow();

	    //println!("{:?}", grid);
	    let args = vec![ cuda::execute::LaunchKernelArg::DeviceMemory(&z_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&x_devmem),
			     cuda::execute::LaunchKernelArg::DeviceMemory(&y_devmem),
			     cuda::execute::LaunchKernelArg::Int(x.shape()[0] as i32),
			     cuda::execute::LaunchKernelArg::Int(x.shape()[1] as i32),
			     cuda::execute::LaunchKernelArg::Int(y.shape()[1] as i32)];
	    let result = cuda::execute::launch_kernel(&self.matmul_func.1,
						      grid,
						      block,
						      args);

	    if let Err(err) = result {
		return Err(err);
	    }
	}
	Ok(z_tensor)
    }
}

