use crate::*;
use std::os::raw::*;

pub fn max_potential_blocksize(function : &cuda::module::Function,
			       dynamic_smem_size: usize,
			       block_size_limit: i32) ->
    Result<(i32 /*minGridSize*/, i32 /* blockSize */), CUDAError>
{
    let mut min_grid_size:i32 = 0;
    let mut block_size:i32 = 0;
    let result:CUDAError = unsafe {
	cuOccupancyMaxPotentialBlockSize(&mut min_grid_size as *mut i32,
					 &mut block_size as *mut i32,
					 function.as_raw(),
					 None,
					 dynamic_smem_size as u64,
					 block_size_limit)
    }.into();
    if result != CUDAError::Success {
	return Err(result);
    }
    Ok((min_grid_size,block_size))
}
