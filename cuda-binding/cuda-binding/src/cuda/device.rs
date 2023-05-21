use crate::*;
use std::mem;
use uuid::Uuid;

pub struct Props {
    name: String,
    uuid: Uuid,
    luid: [u8;8],
    luid_device_node_mask: u32,
    total_global_mem: usize,
    shared_mem_per_block: usize,
    regs_per_block: i32,
    wrap_size: i32,
    mem_pitch: usize,
    max_threads_per_block: usize,
    max_thread_dim: [i32;3],
    max_grid_size: [i32;3],
    total_const_mem: size_t,
    major: i32,
    minor: i32,
    texture_alignment: usize,
    multi_processor_count: i32,
    integrated: bool,
    can_map_host_memory: bool,
    max_texture_1d: i32,
    max_texture_1d_mipmap: i32,
    max_texture_2d: [i32;2],
    max_texture_2d_mipmap: [i32;2],
    max_texture_2d_linear: [i32;3],
    max_texture_2d_gather: [i32;2],
    max_texture_3d: [i32;3],
    max_texture_3d_alt: [i32;3],
    max_texture_cubemap: i32,
    max_texture_1d_layered: [i32;2],
    max_texture_2d_layered: [i32;3],
    max_texture_cubmap_layered: [i32;2],
    max_surface_1d: i32,
    max_surface_2d: [i32;2],
    max_surface_3d: [i32;3],
    max_surface_1d_layered: [i32;2],
    max_surface_2d_layered: [i32;3],
    max_surface_cubemap: i32,
    max_surface_cubemap_layered: [i32;2],
    surface_alignment: usize,
    concurrent_kernels: i32,
    ecc_enabled: bool,
    pci_bus_id: i32,
    pci_device_id: i32,
    pci_domain_id: i32,
    tcc_driver: i32,
    async_engine_count: i32,
    unified_addressing: i32,
    memory_bus_width: i32,
    l2_cache_size: i32,
    persisting_l2_cache_max_size: i32,
    max_threads_per_multi_processor: i32,
    stream_priorities_supported: i32,
    global_l1_cache_supported: i32,
    local_l1_cache_supported: i32,
    shared_mem_per_multiprocessor: usize,
    regs_per_multiprocessor: i32,
    managed_memory: bool,
    is_multi_gpu_board: bool,
    multi_gpu_board_group_id: i32,
    host_native_atomic_supported: bool,
    pageable_memory_access: bool,
    concurrent_managed_access: bool,
    compute_preemption_supported: bool,
    can_use_host_pointer_for_registered_mem: bool,
    cooperative_launch: bool,
    shared_mem_per_block_optin: usize,
    pageable_memory_access_uses_host_page_tables: bool,
    direct_managed_mem_access_from_host: bool,
    max_blocks_per_multi_processor: i32,
    access_policy_max_window_size: i32,
    reserved_shared_mem_per_block: usize,
    host_register_supported: bool,
    sparse_cuda_array_supported: bool,
    host_register_read_only_supported: bool,
    timeline_semaphore_interop_supported: bool,
    memory_pools_supported: bool,
    gpu_direct_rdma_supported: bool,
    gpu_direct_rdma_flush_writes_options: u32,
    gpu_direct_rdma_writes_ordering: i32,
    memory_pool_supported_handle_types: u32,
    deferred_mapping_cuda_array_supported: bool,
    ipc_event_supported: bool,
    cluster_launch: bool,
    unified_function_pointers: bool,
}

impl Props {

    pub fn name(&self) -> &str {
	&self.name
    }

    pub fn shared_mem_per_block(&self) -> usize {
	self.shared_mem_per_block
    }

    pub fn max_threads_per_block(&self) -> usize {
	self.max_threads_per_block
    }
}

pub struct Device {
    device: CUdevice
}

impl Device {

    pub fn get_count() -> Result<i32, CUDAError> {
	let mut count:i32 = 0;
	let result:CUDAError = unsafe { cuDeviceGetCount(&mut count) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(count)
    }

    pub fn new(deviceNo:i32) -> Result<Device, CUDAError> {
	let mut device:CUdevice = 0;

	let result:CUDAError = unsafe { cuDeviceGet(&mut device, deviceNo) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(Device {
	    device
	})
    }

    pub fn as_raw(&self) -> i32 {
	self.device
    }

    pub fn get_compute_capability(&self) -> Result<(u32,u32),CUDAError> {
	let (mut major, mut minor):(i32,i32) = (0,0);
	let result:CUDAError = unsafe {
	    cuDeviceGetAttribute(&mut major,
				 CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
				 self.device) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	let result:CUDAError = unsafe {
	    cuDeviceGetAttribute(&mut minor,
				 CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
				 self.device) }.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok((major as u32, minor as u32))
    }

    pub fn get_name(&self) -> Result<String, CUDAError> {
	let mut name:[i8;100] = [0;100];
	let result:CUDAError = unsafe {
	    cuDeviceGetName(name.as_mut_ptr(), 100, self.device)
	}.into();
	if result != CUDAError::Success {
	    return Err(result);
	}
	Ok(String::from_utf8(name.into_iter().map(|c| c as u8).collect()).unwrap())
    }

    pub fn get_properites(&self) -> Result<Props,CUDAError> {

	let (result, cuda_props):(CUDAError, cudaDeviceProp) = unsafe {
	    let mut cuda_props:cudaDeviceProp = mem::MaybeUninit::zeroed().assume_init();
	    let result = cudaGetDeviceProperties_v2(&mut cuda_props,
						    self.device);
	    (result.into(), cuda_props)
	};

	if result != CUDAError::Success {
	    return Err(result);
	}
	let props = Props {
	    name: String::from_utf8(cuda_props.name.iter().map(|&c| c as u8).collect()).unwrap(),
	    uuid: uuid::Builder::from_slice(&(cuda_props.uuid.bytes.iter().map(|&c| c as u8).collect::<Vec<u8>>()))
		.unwrap().into_uuid(),
	    luid: [cuda_props.luid[0] as u8,
		   cuda_props.luid[1] as u8,
	    	   cuda_props.luid[2] as u8,
		   cuda_props.luid[3] as u8,
		   cuda_props.luid[4] as u8,
		   cuda_props.luid[5] as u8,
	    	   cuda_props.luid[6] as u8,
		   cuda_props.luid[7] as u8],
	    luid_device_node_mask: cuda_props.luidDeviceNodeMask,
	    total_global_mem: cuda_props.totalGlobalMem as usize,
	    shared_mem_per_block: cuda_props.sharedMemPerBlock as usize,
	    regs_per_block: cuda_props.regsPerBlock,
	    wrap_size: cuda_props.warpSize,
	    mem_pitch: cuda_props.memPitch as usize,
	    max_threads_per_block: cuda_props.maxThreadsPerBlock as usize,
	    max_thread_dim: cuda_props.maxThreadsDim,
	    max_grid_size: cuda_props.maxGridSize,
	    total_const_mem: cuda_props.totalConstMem,
	    major: cuda_props.major,
	    minor: cuda_props.minor,
	    texture_alignment: cuda_props.textureAlignment as usize,
	    multi_processor_count: cuda_props.multiProcessorCount,
	    integrated: if cuda_props.integrated == 0 { false } else { true },
	    can_map_host_memory: if cuda_props.canMapHostMemory == 0 { false } else { true },
	    max_texture_1d: cuda_props.maxTexture1D,
	    max_texture_1d_mipmap: cuda_props.maxTexture1DMipmap,
	    max_texture_2d: cuda_props.maxTexture2D,
	    max_texture_2d_mipmap: cuda_props.maxTexture2DMipmap,
	    max_texture_2d_linear: cuda_props.maxTexture2DLinear,
	    max_texture_2d_gather: cuda_props.maxTexture2DGather,
	    max_texture_3d: cuda_props.maxTexture3D,
	    max_texture_3d_alt: cuda_props.maxTexture3DAlt,
	    max_texture_cubemap: cuda_props.maxTextureCubemap,
	    max_texture_1d_layered: cuda_props.maxTexture1DLayered,
	    max_texture_2d_layered: cuda_props.maxTexture2DLayered,
	    max_texture_cubmap_layered: cuda_props.maxTextureCubemapLayered,
	    max_surface_1d: cuda_props.maxSurface1D,
	    max_surface_2d: cuda_props.maxSurface2D,
	    max_surface_3d: cuda_props.maxSurface3D,
	    max_surface_1d_layered: cuda_props.maxSurface1DLayered,
	    max_surface_2d_layered: cuda_props.maxSurface2DLayered,
	    max_surface_cubemap: cuda_props.maxSurfaceCubemap,
	    max_surface_cubemap_layered: cuda_props.maxSurfaceCubemapLayered,
	    surface_alignment: cuda_props.surfaceAlignment as usize,
	    concurrent_kernels: cuda_props.concurrentKernels,
	    ecc_enabled: if cuda_props.ECCEnabled == 0 { false } else { true },
	    pci_bus_id: cuda_props.pciBusID,
	    pci_device_id: cuda_props.pciDeviceID,
	    pci_domain_id: cuda_props.pciDomainID,
	    tcc_driver: cuda_props.tccDriver,
	    async_engine_count: cuda_props.asyncEngineCount,
	    unified_addressing: cuda_props.unifiedAddressing,
	    memory_bus_width: cuda_props.memoryBusWidth,
	    l2_cache_size: cuda_props.l2CacheSize,
	    persisting_l2_cache_max_size: cuda_props.persistingL2CacheMaxSize,
	    max_threads_per_multi_processor: cuda_props.maxThreadsPerMultiProcessor,
	    stream_priorities_supported: cuda_props.streamPrioritiesSupported,
	    global_l1_cache_supported: cuda_props.globalL1CacheSupported,
	    local_l1_cache_supported: cuda_props.localL1CacheSupported,
	    shared_mem_per_multiprocessor: cuda_props.sharedMemPerMultiprocessor as usize,
	    regs_per_multiprocessor: cuda_props.regsPerMultiprocessor,
	    managed_memory: if cuda_props.managedMemory == 0 { false } else { true },
	    is_multi_gpu_board: if cuda_props.isMultiGpuBoard == 0 { false } else { true },
	    multi_gpu_board_group_id: cuda_props.multiGpuBoardGroupID,
	    host_native_atomic_supported: !(cuda_props.hostNativeAtomicSupported == 0),
	    pageable_memory_access: !(cuda_props.pageableMemoryAccess == 0),
	    concurrent_managed_access: !(cuda_props.concurrentManagedAccess == 0),
	    compute_preemption_supported: !(cuda_props.computePreemptionSupported == 0),
	    can_use_host_pointer_for_registered_mem: !(cuda_props.canUseHostPointerForRegisteredMem == 0),
	    cooperative_launch: !(cuda_props.cooperativeLaunch == 0),
	    shared_mem_per_block_optin: cuda_props.sharedMemPerBlockOptin as usize,
	    pageable_memory_access_uses_host_page_tables: !(cuda_props.pageableMemoryAccessUsesHostPageTables == 0),
	    direct_managed_mem_access_from_host: !(cuda_props.directManagedMemAccessFromHost == 0),
	    max_blocks_per_multi_processor: cuda_props.maxBlocksPerMultiProcessor,
	    access_policy_max_window_size: cuda_props.accessPolicyMaxWindowSize,
	    reserved_shared_mem_per_block: cuda_props.reservedSharedMemPerBlock as usize,
	    host_register_supported: !(cuda_props.hostRegisterReadOnlySupported == 0),
	    sparse_cuda_array_supported: !(cuda_props.sparseCudaArraySupported == 0),
	    host_register_read_only_supported: !(cuda_props.hostRegisterReadOnlySupported == 0),
	    timeline_semaphore_interop_supported: !(cuda_props.timelineSemaphoreInteropSupported == 0),
	    memory_pools_supported: !(cuda_props.memoryPoolsSupported == 0),
	    gpu_direct_rdma_supported: (0 < cuda_props.gpuDirectRDMASupported),
	    gpu_direct_rdma_flush_writes_options: cuda_props.gpuDirectRDMAFlushWritesOptions,
	    gpu_direct_rdma_writes_ordering: cuda_props.gpuDirectRDMAWritesOrdering,
	    memory_pool_supported_handle_types: cuda_props.memoryPoolSupportedHandleTypes,
	    deferred_mapping_cuda_array_supported: (0 < cuda_props.deferredMappingCudaArraySupported),
	    ipc_event_supported: (0 < cuda_props.ipcEventSupported),
	    cluster_launch: (0 < cuda_props.clusterLaunch),
	    unified_function_pointers: (0 < cuda_props.unifiedFunctionPointers),
	};
	Ok(props)
    }
}

pub fn synchronize() -> Result<(),CUDAError> {
    let result:CUDAError = unsafe {
	cudaDeviceSynchronize()
    }.into();
    if result != CUDAError::Success {
	return Err(result);
    }
    Ok(())
}
