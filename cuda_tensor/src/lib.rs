use cuda_binding::*;
use std::env;
use std::rc::Rc;

pub mod tensor;
use crate::tensor::Tensor;

#[derive(Debug)]
pub struct Context {
    dev_context: cuda::SharableContext,
    add_func: (Rc<cuda::Module>, cuda::Function),
    sub_func: (Rc<cuda::Module>, cuda::Function),
    hadamard_func: (Rc<cuda::Module>, cuda::Function),
}

impl Context {
    pub fn new(dev_context: cuda::SharableContext) -> Result<Context, CUDAError> {
        let fatbins = [include_bytes!(concat!(env!("OUT_DIR"), "/tensor_add.fatbin")).to_vec(),
		       include_bytes!(concat!(env!("OUT_DIR"), "/hadamard_product.fatbin")).to_vec()];

        //dev_context.push()?;
        let module   = {
	    let m = cuda::Module::new(&fatbins[0])?;
	    Rc::new(m)
	};
        let func     = module.get_function("tensor_add")?;
        let add_func = (Rc::clone(&module), func);
        let func     = module.get_function("tensor_sub")?;
	let sub_func = (module, func);
	let module   = {
	    let m = cuda::Module::new(&fatbins[1])?;
	    Rc::new(m)
	};
        let func     = module.get_function("hadamard_product")?;
	let hadamard_func = (module, func);
	//cuda::Context::pop()?;
        Ok(Context {
            dev_context,
            add_func,
	    sub_func,
	    hadamard_func,
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
}

