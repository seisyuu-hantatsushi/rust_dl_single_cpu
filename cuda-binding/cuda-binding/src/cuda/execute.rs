
use crate::*;
use std::os::raw::*;

pub enum LaunchKernelArg<'a, T> {
    DeviceMemory(&'a cuda::Memory<T>),
    SizeT(usize),
    Int(i32)
}

pub fn launch_kernel<'a, T>(function:&cuda::Function,
			    grid:(usize, usize, usize),
			    block:(usize, usize, usize),
			    args:Vec<LaunchKernelArg<'a, T>>) -> Result<(), CUDAError> {

    let mut params_value:Vec<*mut c_void> = vec!();

    for arg in args.iter() {
	match &arg {
	    LaunchKernelArg::DeviceMemory(device_mem) => {
		params_value.push(device_mem.raw_ptr())
	    },
	    LaunchKernelArg::SizeT(a) => {
		let ptr:*mut c_void = unsafe { std::mem::transmute(*a) };
		params_value.push(ptr)
	    },
	    LaunchKernelArg::Int(a) => {
		let ptr:*mut c_void = unsafe { std::mem::transmute(*a as u64) };
		params_value.push(ptr)
	    }
	}
    }

    let mut params_ref:Vec<*mut c_void> = vec!();
    let params_value_ptr = params_value.as_ptr();
    unsafe {
	for i in 0..params_value.len() {
	    //println!("{:p} in {:p}", params_value[i], params_value_ptr.add(i));
	    params_ref.push(params_value_ptr.add(i) as *mut c_void);
	}
    }

    //println!("function handle: {:p}",function.as_raw());
    let result:CUDAError = unsafe {
	cuLaunchKernel(function.as_raw(),
		       grid.0 as u32, grid.1 as u32, grid.2 as u32,
		       block.0 as u32, block.1 as u32, block.2 as u32,
		       0,
		       std::ptr::null_mut(),
		       params_ref.as_ptr() as *mut *mut c_void,
		       std::ptr::null_mut())
    }.into();

    if result != CUDAError::Success {
	eprintln!("launch result {}", result);
	return Err(result);
    }
    //println!("launch kernel");

    Ok(())
}
