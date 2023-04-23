#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::fmt::Display;
use std::convert::{From,Into};
use std::mem::size_of;

pub mod cudnn;

#[derive(Debug)]
pub enum CUDAError {
    Success,
    InvalidValue,
    MemoryAllocation,
    InitializationError,
    CudartUnloading,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    InvalidConfiguration,
    InvalidPitchValue,
    InvalidSymbol,
    InvalidHostPointer,
    InvalidDevicePointer,
    InvalidTexture,
    InvalidTextureBinding,
    InvalidChannelDescriptor,
    InvalidMemcpyDirection,
    AddressOfConstant,
    TextureFetchFailed,
    TextureNotBound,
    SynchronizationError,
    InvalidFilterSetting,
    InvalidNormSetting,
    MixedDeviceExecution,
    NotYetImplemented,
    MemoryValueTooLarge,
    StubLibrary,
    InsufficientDriver,
    CallRequiresNewerDriver,
    InvalidSurface,
    DuplicateVariableName,
    DuplicateTextureName,
    DuplicateSurfaceName,
    DevicesUnavailable,
    IncompatibleDriverContext,
    MissingConfiguration,
    PriorLaunchFailure,
    LaunchMaxDepthExceeded,
    LaunchFileScopedTex,
    LaunchFileScopedSurf,
    SyncDepthExceeded,
    LaunchPendingCountExceeded,
    InvalidDeviceFunction,
    NoDevice,
    InvalidDevice,
    DeviceNotLicensed,
    SoftwareValidityNotEstablished,
    StartupFailure,
    InvalidKernelImage,
    DeviceUninitialized,
    MapBufferObjectFailed,
    UnmapBufferObjectFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoKernelImageForDevice,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    ECCUncorrectable,
    UnsupportedLimit,
    DeviceAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    JitCompilerNotFound,
    UnsupportedPtxVersion,
    JitCompilationDisabled,
    UnsupportedExecAffinity,
    UnsupportedDevSideSync,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidResourceHandle,
    IllegalState,
    SymbolNotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    SetOnActiveProcess,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailure,
    CooperativeLaunchTooLarge,
    NotPermitted,
    NotSupported,
    SystemNotReady,
    SystemDriverMismatch,
    CompatNotSupportedOnDevice,
    MpsConnectionFailed,
    MpsRpcFailure,
    MpsServerNotReady,
    MpsMaxClientsReached,
    MpsMaxConnectionsReached,
    MpsClientTerminated,
    CdpNotSupported,
    CdpVersionMismatch,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    Timeout,
    GraphExecUpdateFailure,
    ExternalDevice,
    InvalidClusterSize,
    Unknown,
    ApiFailureBase
}

impl std::error::Error for CUDAError {}

impl Display for CUDAError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
	use self::CUDAError::*;
	match self {
	    Success                  => write!(f,"CUDAError::Success"),
	    InvalidValue             => write!(f,"CUDAError::InvalidValue"),
	    MemoryAllocation         => write!(f,"CUDAError::MemoryAllocation"),
	    InitializationError      => write!(f,"CUDAError::CudartUnloading"),
	    CudartUnloading          => write!(f,"CUDAError::CudartUnloading"),
	    ProfilerDisabled         => write!(f,"CUDAError::ProfilerDisabled"),
	    ProfilerNotInitialized   => write!(f,"CUDAError::ProfilerNotInitialized"),
	    ProfilerAlreadyStarted   => write!(f,"CUDAError::ProfilerAlreadyStarted"),
	    ProfilerAlreadyStopped   => write!(f,"CUDAError::ProfilerAlreadyStopped"),
	    InvalidConfiguration     => write!(f,"CUDAError::InvalidConfiguration"),
	    InvalidPitchValue        => write!(f,"CUDAError::InvalidPitchValue"),
	    InvalidSymbol            => write!(f,"CUDAError::InvalidSymbol"),
	    InvalidHostPointer       => write!(f,"CUDAError::InvalidHostPointer"),
	    InvalidDevicePointer     => write!(f,"CUDAError::InvalidDevicePointer"),
	    InvalidTexture           => write!(f,"CUDAError::InvalidTexture"),
	    InvalidTextureBinding    => write!(f,"CUDAError::InvalidTextureBinding"),
	    InvalidChannelDescriptor => write!(f,"CUDAError::InvalidChannelDescriptor"),
	    InvalidMemcpyDirection   => write!(f,"CUDAError::InvalidMemcpyDirection"),
	    AddressOfConstant        => write!(f,"CUDAError::AddressOfConstant"),
	    TextureFetchFailed       => write!(f,"CUDAError::TextureFetchFailed"),
	    TextureNotBound          => write!(f,"CUDAError::TextureNotBound"),
	    SynchronizationError     => write!(f,"CUDAError::SynchronizationError"),
	    InvalidFilterSetting     => write!(f,"CUDAError::InvalidFilterSetting"),
	    InvalidNormSetting       => write!(f,"CUDAError::InvalidNormSetting"),
	    MixedDeviceExecution     => write!(f,"CUDAError::MixedDeviceExecution"),
	    NotYetImplemented        => write!(f,"CUDAError::NotYetImplemented"),
	    MemoryValueTooLarge      => write!(f,"CUDAError::MemoryValueTooLarge"),
	    StubLibrary              => write!(f,"CUDAError::StubLibrary"),
	    InsufficientDriver       => write!(f,"CUDAError::InsufficientDriver"),
	    CallRequiresNewerDriver  => write!(f,"CUDAError::CallRequiresNewerDriver"),
	    InvalidSurface           => write!(f,"CUDAError::InvalidSurface"),
	    DuplicateVariableName    => write!(f,"CUDAError::DuplicateVariableName"),
	    DuplicateTextureName     => write!(f,"CUDAError::DuplicateTextureName"),
	    DuplicateSurfaceName     => write!(f,"CUDAError::DuplicateSurfaceName"),
	    DevicesUnavailable       => write!(f,"CUDAError::DevicesUnavailable"),
	    IncompatibleDriverContext => write!(f,"CUDAError::IncompatibleDriverContext"),
	    MissingConfiguration  => write!(f,"CUDAError::MissingConfiguration"),
	    PriorLaunchFailure  => write!(f,"CUDAError::PriorLaunchFailure"),
	    LaunchMaxDepthExceeded  => write!(f,"CUDAError::LaunchMaxDepthExceeded"),
	    LaunchFileScopedTex  => write!(f,"CUDAError::LaunchFileScopedTex"),
	    LaunchFileScopedSurf  => write!(f,"CUDAError::LaunchFileScopedSurf"),
	    SyncDepthExceeded  => write!(f,"CUDAError::SyncDepthExceeded"),
	    LaunchPendingCountExceeded  => write!(f,"CUDAError::LaunchPendingCountExceeded"),
	    InvalidDeviceFunction  => write!(f,"CUDAError::InvalidDeviceFunction"),
	    NoDevice  => write!(f,"CUDAError::NoDevice"),
	    InvalidDevice  => write!(f,"CUDAError::InvalidDevice"),
	    DeviceNotLicensed  => write!(f,"CUDAError::DeviceNotLicensed"),
	    SoftwareValidityNotEstablished  => write!(f,"CUDAError::SoftwareValidityNotEstablished"),
	    StartupFailure  => write!(f,"CUDAError::StartupFailure"),
	    InvalidKernelImage  => write!(f,"CUDAError::InvalidKernelImage"),
	    DeviceUninitialized  => write!(f,"CUDAError::DeviceUninitialized"),
	    MapBufferObjectFailed  => write!(f,"CUDAError::MapBufferObjectFailed"),
	    UnmapBufferObjectFailed  => write!(f,"CUDAError::UnmapBufferObjectFailed"),
	    ArrayIsMapped  => write!(f,"CUDAError::ArrayIsMapped"),
	    AlreadyMapped  => write!(f,"CUDAError::AlreadyMapped"),
	    NoKernelImageForDevice  => write!(f,"CUDAError::NoKernelImageForDevice"),
	    AlreadyAcquired  => write!(f,"CUDAError::AlreadyAcquired"),
	    NotMapped  => write!(f,"CUDAError::NotMapped"),
	    NotMappedAsArray  => write!(f,"CUDAError::NotMappedAsArray"),
	    NotMappedAsPointer  => write!(f,"CUDAError::NotMappedAsPointer"),
	    ECCUncorrectable  => write!(f,"CUDAError::ECCUncorrectable"),
	    UnsupportedLimit  => write!(f,"CUDAError::UnsupportedLimit"),
	    DeviceAlreadyInUse  => write!(f,"CUDAError::DeviceAlreadyInUse"),
	    PeerAccessUnsupported  => write!(f,"CUDAError::PeerAccessUnsupported"),
	    InvalidPtx  => write!(f,"CUDAError::InvalidPtx"),
	    InvalidGraphicsContext  => write!(f,"CUDAError::InvalidGraphicsContext"),
	    NvlinkUncorrectable  => write!(f,"CUDAError::NvlinkUncorrectable"),
	    JitCompilerNotFound  => write!(f,"CUDAError::JitCompilerNotFound"),
	    UnsupportedPtxVersion => write!(f,"CUDAError:: UnsupportedPtxVersion"),
	    JitCompilationDisabled => write!(f,"CUDAError::JitCompilationDisabled"),
	    UnsupportedExecAffinity => write!(f,"CUDAError::UnsupportedExecAffinity"),
	    UnsupportedDevSideSync => write!(f,"CUDAError::UnsupportedDevSideSync"),
	    InvalidSource => write!(f,"CUDAError::InvalidSource"),
	    FileNotFound => write!(f,"CUDAError::FileNotFound"),
	    SharedObjectSymbolNotFound => write!(f,"CUDAError::SharedObjectSymbolNotFound"),
	    SharedObjectInitFailed => write!(f,"CUDAError::SharedObjectInitFailed"),
	    OperatingSystem => write!(f,"CUDAError::OperatingSystem"),
	    InvalidResourceHandle => write!(f,"CUDAError::InvalidResourceHandle"),
	    IllegalState => write!(f,"CUDAError::IllegalState"),
	    SymbolNotFound => write!(f,"CUDAError::SymbolNotFound"),
	    NotReady => write!(f,"CUDAError::NotReady"),
	    IllegalAddress => write!(f,"CUDAError::IllegalAddress"),
	    LaunchOutOfResources => write!(f,"CUDAError::LaunchOutOfResources"),
	    LaunchTimeout => write!(f,"CUDAError::LaunchTimeout"),
	    LaunchIncompatibleTexturing => write!(f,"CUDAError::LaunchIncompatibleTexturing"),
	    PeerAccessAlreadyEnabled => write!(f,"CUDAError::PeerAccessAlreadyEnabled"),
	    PeerAccessNotEnabled => write!(f,"CUDAError::PeerAccessNotEnabled"),
	    SetOnActiveProcess => write!(f,"CUDAError::SetOnActiveProcess"),
	    ContextIsDestroyed => write!(f,"CUDAError::ContextIsDestroyed"),
	    Assert => write!(f,"CUDAError::Assert"),
	    TooManyPeers => write!(f,"CUDAError::TooManyPeers"),
	    HostMemoryAlreadyRegistered => write!(f,"CUDAError::HostMemoryAlreadyRegistered"),
	    HostMemoryNotRegistered => write!(f,"CUDAError::HostMemoryNotRegistered"),
	    HardwareStackError => write!(f,"CUDAError::HardwareStackError"),
	    IllegalInstruction => write!(f,"CUDAError::IllegalInstruction"),
	    MisalignedAddress => write!(f,"CUDAError::MisalignedAddress"),
	    InvalidAddressSpace => write!(f,"CUDAError::InvalidAddressSpace"),
	    InvalidPc => write!(f,"CUDAError::InvalidPc"),
	    LaunchFailure => write!(f,"CUDAError::LaunchFailure"),
	    CooperativeLaunchTooLarge => write!(f,"CUDAError::CooperativeLaunchTooLarge"),
	    NotPermitted => write!(f,"CUDAError::NotPermitted"),
	    NotSupported => write!(f,"CUDAError::NotSupported"),
	    SystemNotReady => write!(f,"CUDAError::SystemNotReady"),
	    SystemDriverMismatch => write!(f,"CUDAError::SystemDriverMismatch"),
	    CompatNotSupportedOnDevice => write!(f,"CUDAError::CompatNotSupportedOnDevice"),
	    MpsConnectionFailed => write!(f,"CUDAError::MpsConnectionFailed"),
	    MpsRpcFailure => write!(f,"CUDAError::MpsRpcFailure"),
	    MpsServerNotReady => write!(f,"CUDAError::MpsServerNotReady"),
	    MpsMaxClientsReached => write!(f,"CUDAError::MpsMaxClientsReached"),
	    MpsMaxConnectionsReached => write!(f,"CUDAError::MpsMaxConnectionsReached"),
	    MpsClientTerminated => write!(f,"CUDAError::MpsClientTerminated"),
	    CdpNotSupported => write!(f,"CUDAError::CdpNotSupported"),
	    CdpVersionMismatch => write!(f,"CUDAError::CdpVersionMismatch"),
	    StreamCaptureUnsupported => write!(f,"CUDAError::StreamCaptureUnsupported"),
	    StreamCaptureInvalidated => write!(f,"CUDAError::StreamCaptureInvalidated"),
	    StreamCaptureMerge => write!(f,"CUDAError::StreamCaptureMerge"),
	    StreamCaptureUnmatched => write!(f,"CUDAError::StreamCaptureUnmatched"),
	    StreamCaptureUnjoined => write!(f,"CUDAError::StreamCaptureUnjoined"),
	    StreamCaptureIsolation => write!(f,"CUDAError::StreamCaptureIsolation"),
	    StreamCaptureImplicit => write!(f,"CUDAError::StreamCaptureImplicit"),
	    CapturedEvent => write!(f,"CUDAError::CapturedEvent"),
	    StreamCaptureWrongThread => write!(f,"CUDAError::StreamCaptureWrongThread"),
	    Timeout => write!(f,"CUDAError::Timeout"),
	    GraphExecUpdateFailure => write!(f,"CUDAError::GraphExecUpdateFailure"),
	    ExternalDevice => write!(f,"CUDAError::ExternalDevice"),
	    InvalidClusterSize => write!(f,"CUDAError::InvalidClusterSize"),
	    Unknown => write!(f,"CUDAError::Unknown"),
	    ApiFailureBase => write!(f,"CUDAError::ApiFailureBase")
	}
    }
}

impl From<cudaError> for CUDAError {
    fn from(error:cudaError) -> Self {
	match error {
	    cudaError_cudaSuccess => CUDAError::Success,
	    cudaError_cudaErrorInvalidValue => CUDAError::InvalidValue,
	    cudaError_cudaErrorMemoryAllocation => CUDAError::MemoryAllocation,
	    cudaError_cudaErrorInitializationError => CUDAError::InitializationError,
	    cudaError_cudaErrorCudartUnloading => CUDAError::CudartUnloading,
	    cudaError_cudaErrorProfilerDisabled => CUDAError::ProfilerDisabled,
	    cudaError_cudaErrorProfilerNotInitialized => CUDAError::ProfilerNotInitialized,
	    cudaError_cudaErrorProfilerAlreadyStarted => CUDAError::ProfilerAlreadyStarted,
	    cudaError_cudaErrorProfilerAlreadyStopped => CUDAError::ProfilerAlreadyStopped,
	    cudaError_cudaErrorInvalidConfiguration => CUDAError::InvalidConfiguration,
	    cudaError_cudaErrorInvalidPitchValue => CUDAError::InvalidPitchValue,
	    cudaError_cudaErrorInvalidSymbol => CUDAError::InvalidSymbol,
	    cudaError_cudaErrorInvalidHostPointer => CUDAError::InvalidHostPointer,
	    cudaError_cudaErrorInvalidDevicePointer => CUDAError::InvalidDevicePointer,
	    cudaError_cudaErrorInvalidTexture => CUDAError::InvalidTexture,
	    cudaError_cudaErrorInvalidTextureBinding => CUDAError::InvalidTextureBinding,
	    cudaError_cudaErrorInvalidChannelDescriptor => CUDAError::InvalidChannelDescriptor,
	    cudaError_cudaErrorInvalidMemcpyDirection => CUDAError::InvalidMemcpyDirection,
	    cudaError_cudaErrorAddressOfConstant => CUDAError::AddressOfConstant,
	    cudaError_cudaErrorTextureFetchFailed => CUDAError::TextureFetchFailed,
	    cudaError_cudaErrorTextureNotBound => CUDAError::TextureNotBound,
	    cudaError_cudaErrorSynchronizationError => CUDAError::SynchronizationError,
	    cudaError_cudaErrorInvalidFilterSetting => CUDAError::InvalidFilterSetting,
	    cudaError_cudaErrorInvalidNormSetting => CUDAError::InvalidNormSetting,
	    cudaError_cudaErrorMixedDeviceExecution => CUDAError::MixedDeviceExecution,
	    cudaError_cudaErrorNotYetImplemented => CUDAError::NotYetImplemented,
	    cudaError_cudaErrorMemoryValueTooLarge => CUDAError::MemoryValueTooLarge,
	    cudaError_cudaErrorStubLibrary => CUDAError::StubLibrary,
	    cudaError_cudaErrorInsufficientDriver => CUDAError::InsufficientDriver,
	    cudaError_cudaErrorCallRequiresNewerDriver => CUDAError::CallRequiresNewerDriver,
	    cudaError_cudaErrorInvalidSurface => CUDAError::InvalidSurface,
	    cudaError_cudaErrorDuplicateVariableName => CUDAError::DuplicateVariableName,
	    cudaError_cudaErrorDuplicateTextureName => CUDAError::DuplicateTextureName,
	    cudaError_cudaErrorDuplicateSurfaceName => CUDAError::DuplicateSurfaceName,
	    cudaError_cudaErrorDevicesUnavailable => CUDAError::DevicesUnavailable,
	    cudaError_cudaErrorIncompatibleDriverContext => CUDAError::IncompatibleDriverContext,
	    cudaError_cudaErrorMissingConfiguration => CUDAError::MissingConfiguration,
	    cudaError_cudaErrorPriorLaunchFailure => CUDAError::PriorLaunchFailure,
	    cudaError_cudaErrorLaunchMaxDepthExceeded => CUDAError::LaunchMaxDepthExceeded,
	    cudaError_cudaErrorLaunchFileScopedTex => CUDAError::LaunchFileScopedTex,
	    cudaError_cudaErrorLaunchFileScopedSurf => CUDAError::LaunchFileScopedSurf,
	    cudaError_cudaErrorSyncDepthExceeded => CUDAError::SyncDepthExceeded,
	    cudaError_cudaErrorLaunchPendingCountExceeded => CUDAError::LaunchPendingCountExceeded,
	    cudaError_cudaErrorInvalidDeviceFunction => CUDAError::InvalidDeviceFunction,
	    cudaError_cudaErrorNoDevice => CUDAError::NoDevice,
	    cudaError_cudaErrorInvalidDevice => CUDAError::InvalidDevice,
	    cudaError_cudaErrorDeviceNotLicensed => CUDAError::DeviceNotLicensed,
	    cudaError_cudaErrorSoftwareValidityNotEstablished => CUDAError::SoftwareValidityNotEstablished,
	    cudaError_cudaErrorStartupFailure => CUDAError::StartupFailure,
	    cudaError_cudaErrorInvalidKernelImage => CUDAError::InvalidKernelImage,
	    cudaError_cudaErrorDeviceUninitialized => CUDAError::DeviceUninitialized,
	    cudaError_cudaErrorMapBufferObjectFailed => CUDAError::MapBufferObjectFailed,
	    cudaError_cudaErrorUnmapBufferObjectFailed => CUDAError::UnmapBufferObjectFailed,
	    cudaError_cudaErrorArrayIsMapped => CUDAError::ArrayIsMapped,
	    cudaError_cudaErrorAlreadyMapped => CUDAError::AlreadyMapped,
	    cudaError_cudaErrorNoKernelImageForDevice => CUDAError::NoKernelImageForDevice,
	    cudaError_cudaErrorAlreadyAcquired => CUDAError::AlreadyAcquired,
	    cudaError_cudaErrorNotMapped => CUDAError::NotMapped,
	    cudaError_cudaErrorNotMappedAsArray => CUDAError::NotMappedAsArray,
	    cudaError_cudaErrorNotMappedAsPointer => CUDAError::NotMappedAsPointer,
	    cudaError_cudaErrorECCUncorrectable => CUDAError::ECCUncorrectable,
	    cudaError_cudaErrorUnsupportedLimit => CUDAError::UnsupportedLimit,
	    cudaError_cudaErrorDeviceAlreadyInUse => CUDAError::DeviceAlreadyInUse,
	    cudaError_cudaErrorPeerAccessUnsupported => CUDAError::PeerAccessUnsupported,
	    cudaError_cudaErrorInvalidPtx => CUDAError::InvalidPtx,
	    cudaError_cudaErrorInvalidGraphicsContext => CUDAError::InvalidGraphicsContext,
	    cudaError_cudaErrorNvlinkUncorrectable => CUDAError::NvlinkUncorrectable,
	    cudaError_cudaErrorJitCompilerNotFound => CUDAError::JitCompilerNotFound,
	    cudaError_cudaErrorUnsupportedPtxVersion => CUDAError::UnsupportedPtxVersion,
	    cudaError_cudaErrorJitCompilationDisabled => CUDAError::JitCompilationDisabled,
	    cudaError_cudaErrorUnsupportedExecAffinity => CUDAError::UnsupportedExecAffinity,
	    cudaError_cudaErrorUnsupportedDevSideSync => CUDAError::UnsupportedDevSideSync,
	    cudaError_cudaErrorInvalidSource => CUDAError::InvalidSource,
	    cudaError_cudaErrorFileNotFound => CUDAError::FileNotFound,
	    cudaError_cudaErrorSharedObjectSymbolNotFound => CUDAError::SharedObjectSymbolNotFound,
	    cudaError_cudaErrorSharedObjectInitFailed => CUDAError::SharedObjectInitFailed,
	    cudaError_cudaErrorOperatingSystem => CUDAError::OperatingSystem,
	    cudaError_cudaErrorInvalidResourceHandle => CUDAError::InvalidResourceHandle,
	    cudaError_cudaErrorIllegalState => CUDAError::IllegalState,
	    cudaError_cudaErrorSymbolNotFound => CUDAError::SymbolNotFound,
	    cudaError_cudaErrorNotReady => CUDAError::NotReady,
	    cudaError_cudaErrorIllegalAddress => CUDAError::IllegalAddress,
	    cudaError_cudaErrorLaunchOutOfResources => CUDAError::LaunchOutOfResources,
	    cudaError_cudaErrorLaunchTimeout => CUDAError::LaunchTimeout,
	    cudaError_cudaErrorLaunchIncompatibleTexturing => CUDAError::LaunchIncompatibleTexturing,
	    cudaError_cudaErrorPeerAccessAlreadyEnabled => CUDAError::PeerAccessAlreadyEnabled,
	    cudaError_cudaErrorPeerAccessNotEnabled => CUDAError::PeerAccessNotEnabled,
	    cudaError_cudaErrorSetOnActiveProcess => CUDAError::SetOnActiveProcess,
	    cudaError_cudaErrorContextIsDestroyed => CUDAError::SetOnActiveProcess,
	    cudaError_cudaErrorAssert => CUDAError::Assert,
	    cudaError_cudaErrorTooManyPeers => CUDAError::TooManyPeers,
	    cudaError_cudaErrorHostMemoryAlreadyRegistered => CUDAError::HostMemoryAlreadyRegistered,
	    cudaError_cudaErrorHostMemoryNotRegistered => CUDAError::HostMemoryNotRegistered,
	    cudaError_cudaErrorHardwareStackError => CUDAError::HardwareStackError,
	    cudaError_cudaErrorIllegalInstruction => CUDAError::IllegalInstruction,
	    cudaError_cudaErrorMisalignedAddress => CUDAError::MisalignedAddress,
	    cudaError_cudaErrorInvalidAddressSpace => CUDAError::InvalidAddressSpace,
	    cudaError_cudaErrorInvalidPc => CUDAError::InvalidPc,
	    cudaError_cudaErrorLaunchFailure => CUDAError::LaunchFailure,
	    cudaError_cudaErrorCooperativeLaunchTooLarge => CUDAError::CooperativeLaunchTooLarge,
	    cudaError_cudaErrorNotPermitted => CUDAError::NotPermitted,
	    cudaError_cudaErrorNotSupported => CUDAError::NotSupported,
	    cudaError_cudaErrorSystemNotReady => CUDAError::SystemNotReady,
	    cudaError_cudaErrorSystemDriverMismatch => CUDAError::SystemDriverMismatch,
	    cudaError_cudaErrorCompatNotSupportedOnDevice => CUDAError::CompatNotSupportedOnDevice,
	    cudaError_cudaErrorMpsConnectionFailed => CUDAError::MpsConnectionFailed,
	    cudaError_cudaErrorMpsRpcFailure => CUDAError::MpsRpcFailure,
	    cudaError_cudaErrorMpsServerNotReady => CUDAError::MpsServerNotReady,
	    cudaError_cudaErrorMpsMaxClientsReached => CUDAError::MpsMaxClientsReached,
	    cudaError_cudaErrorMpsMaxConnectionsReached => CUDAError::MpsMaxConnectionsReached,
	    cudaError_cudaErrorMpsClientTerminated => CUDAError::MpsClientTerminated,
	    cudaError_cudaErrorCdpNotSupported => CUDAError::CdpNotSupported,
	    cudaError_cudaErrorCdpVersionMismatch => CUDAError::CdpVersionMismatch,
	    cudaError_cudaErrorStreamCaptureUnsupported => CUDAError::StreamCaptureUnsupported,
	    cudaError_cudaErrorStreamCaptureInvalidated => CUDAError::StreamCaptureInvalidated,
	    cudaError_cudaErrorStreamCaptureMerge => CUDAError::StreamCaptureMerge,
	    cudaError_cudaErrorStreamCaptureUnmatched => CUDAError::StreamCaptureUnmatched,
	    cudaError_cudaErrorStreamCaptureUnjoined => CUDAError::StreamCaptureUnjoined,
	    cudaError_cudaErrorStreamCaptureIsolation => CUDAError::StreamCaptureIsolation,
	    cudaError_cudaErrorStreamCaptureImplicit => CUDAError::StreamCaptureImplicit,
	    cudaError_cudaErrorCapturedEvent => CUDAError::CapturedEvent,
	    cudaError_cudaErrorStreamCaptureWrongThread => CUDAError::StreamCaptureWrongThread,
	    cudaError_cudaErrorTimeout => CUDAError::Timeout,
	    cudaError_cudaErrorGraphExecUpdateFailure => CUDAError::GraphExecUpdateFailure,
	    cudaError_cudaErrorExternalDevice => CUDAError::ExternalDevice,
	    cudaError_cudaErrorInvalidClusterSize => CUDAError::InvalidClusterSize,
	    cudaError_cudaErrorUnknown => CUDAError::Unknown,
	    cudaError_cudaErrorApiFailureBase => CUDAError::ApiFailureBase,
	    _ => panic!("unknow datatype {}", error)
	}
    }
}

pub enum CUDAMemcpyKind {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Default,
}

impl From<cudaMemcpyKind> for CUDAMemcpyKind {
    fn from(kind:cudaMemcpyKind) -> Self {
	match kind {
	    cudaMemcpyKind_cudaMemcpyHostToHost => CUDAMemcpyKind::HostToHost,
	    cudaMemcpyKind_cudaMemcpyHostToDevice => CUDAMemcpyKind::HostToDevice,
	    cudaMemcpyKind_cudaMemcpyDeviceToHost => CUDAMemcpyKind::DeviceToHost,
	    cudaMemcpyKind_cudaMemcpyDeviceToDevice => CUDAMemcpyKind::DeviceToDevice,
	    cudaMemcpyKind_cudaMemcpyDefault => CUDAMemcpyKind::Default,
	    _ => panic!("unknown cudaMemcpy Kind {}", kind)
	}
    }
}

impl From<CUDAMemcpyKind> for cudaMemcpyKind {
    fn from(kind:CUDAMemcpyKind) -> Self {
	match kind {
	    CUDAMemcpyKind::HostToHost => cudaMemcpyKind_cudaMemcpyHostToHost,
	    CUDAMemcpyKind::HostToDevice => cudaMemcpyKind_cudaMemcpyHostToDevice,
	    CUDAMemcpyKind::DeviceToHost => cudaMemcpyKind_cudaMemcpyDeviceToHost,
	    CUDAMemcpyKind::DeviceToDevice => cudaMemcpyKind_cudaMemcpyDeviceToDevice,
	    CUDAMemcpyKind::Default => cudaMemcpyKind_cudaMemcpyDefault
	}
    }
}


#[derive(Debug)]
pub enum CUDNNStatus {
    SUCCESS,
    NOT_INITIALIZED,
    ALLOC_FAILED,
    BAD_PARAM,
    INTERNAL_ERROR,
    INVALID_VALUE,
    ARCH_MISMATCH,
    MAPPING_ERROR,
    EXECUTION_FAILED,
    NOT_SUPPORTED,
    LICENSE_ERROR,
    RUNTIME_PREREQUISITE_MISSING,
    RUNTIME_IN_PROGRESS,
    RUNTIME_FP_OVERFLOW,
    VERSION_MISMATCH
}

impl Display for CUDNNStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
	use self::CUDNNStatus::*;
        match self {
	    SUCCESS          => write!(f,"CUDNNStatus::SUCCESS"),
	    NOT_INITIALIZED  => write!(f,"CUDNNStatus::NOT_INITIALIZED"),
	    ALLOC_FAILED     => write!(f,"CUDNNStatus::ALLOC_FAILED"),
	    BAD_PARAM        => write!(f,"CUDNNStatus::BAD_PARAM"),
	    INTERNAL_ERROR   => write!(f,"CUDNNStatus::INTERNAL_ERROR"),
	    INVALID_VALUE    => write!(f,"CUDNNStatus::INVALID_VALUE"),
	    ARCH_MISMATCH    => write!(f,"CUDNNStatus::ARCH_MISMATCH"),
	    MAPPING_ERROR    => write!(f,"CUDNNStatus::MAPPING_ERROR"),
	    EXECUTION_FAILED => write!(f,"CUDNNStatus::EXECUTION_FAILED"),
	    NOT_SUPPORTED    => write!(f,"CUDNNStatus::NOT_SUPPORTED"),
	    LICENSE_ERROR    => write!(f,"CUDNNStatus::LICENSE_ERROR"),
	    RUNTIME_PREREQUISITE_MISSING => write!(f,"CUDNNStatus::RUNTIME_PREREQUISITE_MISSING"),
	    RUNTIME_IN_PROGRESS => write!(f, "CUDNNStatus::RUNTIME_IN_PROGRESS"),
	    RUNTIME_FP_OVERFLOW => write!(f, "CUDNNStatus::RUNTIME_FP_OVERFLOW"),
	    VERSION_MISMATCH => write!(f, "CUDNNStatus::VERSION_MISMATCH")
        }
    }
}

impl std::error::Error for CUDNNStatus {}

impl From<cudnnStatus_t> for CUDNNStatus {
    fn from(status:cudnnStatus_t) -> Self {
	match status {
	    cudnnStatus_t_CUDNN_STATUS_SUCCESS          => CUDNNStatus::SUCCESS,
	    cudnnStatus_t_CUDNN_STATUS_NOT_INITIALIZED  => CUDNNStatus::NOT_INITIALIZED,
	    cudnnStatus_t_CUDNN_STATUS_ALLOC_FAILED     => CUDNNStatus::ALLOC_FAILED,
	    cudnnStatus_t_CUDNN_STATUS_BAD_PARAM        => CUDNNStatus::BAD_PARAM,
	    cudnnStatus_t_CUDNN_STATUS_INTERNAL_ERROR   => CUDNNStatus::INTERNAL_ERROR,
	    cudnnStatus_t_CUDNN_STATUS_INVALID_VALUE    => CUDNNStatus::INVALID_VALUE,
	    cudnnStatus_t_CUDNN_STATUS_ARCH_MISMATCH    => CUDNNStatus::ARCH_MISMATCH,
	    cudnnStatus_t_CUDNN_STATUS_MAPPING_ERROR    => CUDNNStatus::MAPPING_ERROR,
	    cudnnStatus_t_CUDNN_STATUS_EXECUTION_FAILED => CUDNNStatus::EXECUTION_FAILED,
	    cudnnStatus_t_CUDNN_STATUS_NOT_SUPPORTED    => CUDNNStatus::NOT_SUPPORTED,
	    cudnnStatus_t_CUDNN_STATUS_LICENSE_ERROR    => CUDNNStatus::LICENSE_ERROR,
	    cudnnStatus_t_CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => CUDNNStatus::RUNTIME_PREREQUISITE_MISSING,
	    cudnnStatus_t_CUDNN_STATUS_RUNTIME_IN_PROGRESS => CUDNNStatus::RUNTIME_IN_PROGRESS,
	    cudnnStatus_t_CUDNN_STATUS_RUNTIME_FP_OVERFLOW => CUDNNStatus::RUNTIME_FP_OVERFLOW,
	    cudnnStatus_t_CUDNN_STATUS_VERSION_MISMATCH    => CUDNNStatus::VERSION_MISMATCH,
	    _ => panic!("unknow status code {}", status)
	}
    }
}

#[derive(Clone, Copy)]
pub enum CUDNNDataType {
    FLOAT,
    DOUBLE,
    HALF,
    INT8,
    INT32,
    INT8x4,
    UINT8,
    UINT8x4,
    INT8x32,
    BFLOAT16,
    INT64,
    BOOLEAN,
    FP8_E4M3,
    FP8_E5M2,
    FAST_FLOAT_FOR_FP8
}

impl From<cudnnDataType_t> for CUDNNDataType {
    fn from(dataType:cudnnDataType_t) -> Self {
	match dataType {
	    cudnnDataType_t_CUDNN_DATA_FLOAT    => CUDNNDataType::FLOAT,
	    cudnnDataType_t_CUDNN_DATA_DOUBLE   => CUDNNDataType::DOUBLE,
	    cudnnDataType_t_CUDNN_DATA_HALF     => CUDNNDataType::HALF,
	    cudnnDataType_t_CUDNN_DATA_INT8     => CUDNNDataType::INT8,
	    cudnnDataType_t_CUDNN_DATA_INT32    => CUDNNDataType::INT32,
	    cudnnDataType_t_CUDNN_DATA_UINT8    => CUDNNDataType::UINT8,
	    cudnnDataType_t_CUDNN_DATA_INT8x4   => CUDNNDataType::INT8x4,
	    cudnnDataType_t_CUDNN_DATA_UINT8x4  => CUDNNDataType::UINT8x4,
	    cudnnDataType_t_CUDNN_DATA_INT8x32  => CUDNNDataType::INT8x32,
	    cudnnDataType_t_CUDNN_DATA_BFLOAT16 => CUDNNDataType::BFLOAT16,
	    cudnnDataType_t_CUDNN_DATA_INT64    => CUDNNDataType::INT64,
	    cudnnDataType_t_CUDNN_DATA_BOOLEAN  => CUDNNDataType::BOOLEAN,
	    cudnnDataType_t_CUDNN_DATA_FP8_E4M3 => CUDNNDataType::FP8_E4M3,
	    cudnnDataType_t_CUDNN_DATA_FP8_E5M2 => CUDNNDataType::FP8_E5M2,
	    cudnnDataType_t_CUDNN_DATA_FAST_FLOAT_FOR_FP8 => CUDNNDataType::FAST_FLOAT_FOR_FP8,
	    _ => panic!("unknow datatype {}", dataType)
	}
    }
}

impl From<CUDNNDataType> for cudnnDataType_t {
    fn from(dataType:CUDNNDataType) -> Self {
	match dataType {
	    CUDNNDataType::FLOAT  => cudnnDataType_t_CUDNN_DATA_FLOAT,
	    CUDNNDataType::DOUBLE => cudnnDataType_t_CUDNN_DATA_DOUBLE,
	    CUDNNDataType::HALF   => cudnnDataType_t_CUDNN_DATA_HALF,
	    CUDNNDataType::INT8   => cudnnDataType_t_CUDNN_DATA_INT8,
	    CUDNNDataType::INT32  => cudnnDataType_t_CUDNN_DATA_INT32,
	    CUDNNDataType::UINT8  => cudnnDataType_t_CUDNN_DATA_UINT8,
	    CUDNNDataType::INT8x4 => cudnnDataType_t_CUDNN_DATA_INT8x4,
	    CUDNNDataType::UINT8x4 => cudnnDataType_t_CUDNN_DATA_UINT8x4,
	    CUDNNDataType::INT8x32 => cudnnDataType_t_CUDNN_DATA_INT8x32,
	    CUDNNDataType::BFLOAT16 => cudnnDataType_t_CUDNN_DATA_BFLOAT16,
	    CUDNNDataType::INT64    => cudnnDataType_t_CUDNN_DATA_INT64,
	    CUDNNDataType::BOOLEAN  => cudnnDataType_t_CUDNN_DATA_BOOLEAN,
	    CUDNNDataType::FP8_E4M3 => cudnnDataType_t_CUDNN_DATA_FP8_E4M3,
	    CUDNNDataType::FP8_E5M2 => cudnnDataType_t_CUDNN_DATA_FP8_E5M2,
	    CUDNNDataType::FAST_FLOAT_FOR_FP8 => cudnnDataType_t_CUDNN_DATA_FAST_FLOAT_FOR_FP8
	}
    }
}

impl Display for CUDNNDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
	use self::CUDNNDataType::*;
	match self {
	    FLOAT    => write!(f,"CUDNNDataType::FLOAT"),
	    DOUBLE   => write!(f,"CUDNNDataType::DOUBLE"),
	    HALF     => write!(f,"CUDNNDataType::HALF"),
	    INT8     => write!(f,"CUDNNDataType::INT8"),
	    INT32    => write!(f,"CUDNNDataType::INT32"),
	    INT8x4   => write!(f,"CUDNNDataType::INT8x4"),
	    UINT8    => write!(f,"CUDNNDataType::UINT8"),
	    UINT8x4  => write!(f,"CUDNNDataType::UINT8x4"),
	    INT8x32  => write!(f,"CUDNNDataType::INT8x32"),
	    BFLOAT16 => write!(f,"CUDNNDataType::BFLOAT16"),
	    INT64    => write!(f,"CUDNNDataType::INT64"),
	    BOOLEAN  => write!(f,"CUDNNDataType::BOOLEAN"),
	    FP8_E4M3 => write!(f,"CUDNNDataType::FP8_E4M3"),
	    FP8_E5M2 => write!(f,"CUDNNDataType::FP8_E5M2"),
	    FAST_FLOAT_FOR_FP8 => write!(f,"CUDNNDataType::FAST_FLOAT_FOR_FP8")
	}
    }
}

impl CUDNNDataType {
    pub fn size(&self) -> usize {
	use self::CUDNNDataType::*;
	match self {
	    FLOAT    => 4,
	    DOUBLE   => 8,
	    HALF     => 2,
	    INT8     => 1,
	    INT32    => 4,
	    INT8x4   => 4,
	    UINT8    => 1,
	    UINT8x4  => 4,
	    INT8x32  => 32,
	    BFLOAT16 => 2,
	    INT64    => 8,
	    BOOLEAN  => 1,
	    FP8_E4M3 => 1,
	    FP8_E5M2 => 1,
	    FAST_FLOAT_FOR_FP8 => 1
	}
    }
}

pub enum CUDNNTensorOp {
    ADD, MUL, MIN, MAX, SQRT, NOT
}

impl From<cudnnOpTensorOp_t> for CUDNNTensorOp {
    fn from(op:cudnnOpTensorOp_t) -> CUDNNTensorOp {
	match op {
	    cudnnOpTensorOp_t_CUDNN_OP_TENSOR_ADD  => CUDNNTensorOp::ADD,
	    cudnnOpTensorOp_t_CUDNN_OP_TENSOR_MUL  => CUDNNTensorOp::MUL,
	    cudnnOpTensorOp_t_CUDNN_OP_TENSOR_MIN  => CUDNNTensorOp::MIN,
	    cudnnOpTensorOp_t_CUDNN_OP_TENSOR_MAX  => CUDNNTensorOp::MAX,
	    cudnnOpTensorOp_t_CUDNN_OP_TENSOR_SQRT => CUDNNTensorOp::SQRT,
	    cudnnOpTensorOp_t_CUDNN_OP_TENSOR_NOT  => CUDNNTensorOp::NOT,
	    _ => panic!("unknown tensor operator: {}", op)
	}
    }
}

impl From<CUDNNTensorOp> for cudnnOpTensorOp_t {
    fn from(op:CUDNNTensorOp) -> cudnnOpTensorOp_t {
	match op {
	    CUDNNTensorOp::ADD  => cudnnOpTensorOp_t_CUDNN_OP_TENSOR_ADD,
	    CUDNNTensorOp::MUL  => cudnnOpTensorOp_t_CUDNN_OP_TENSOR_MUL,
	    CUDNNTensorOp::MIN  => cudnnOpTensorOp_t_CUDNN_OP_TENSOR_MIN,
	    CUDNNTensorOp::MAX  => cudnnOpTensorOp_t_CUDNN_OP_TENSOR_MAX,
	    CUDNNTensorOp::SQRT => cudnnOpTensorOp_t_CUDNN_OP_TENSOR_SQRT,
	    CUDNNTensorOp::NOT  => cudnnOpTensorOp_t_CUDNN_OP_TENSOR_NOT
	}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;
    use std::os::raw::*;

    #[test]
    fn add_tensor() {
	let mut handle: cudnnHandle_t = ptr::null_mut();
	unsafe {
	    match cudnnCreate(&mut handle) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN context.")
	    }

	    let mut xDesc:cudnnTensorDescriptor_t = ptr::null_mut();
	    let mut yDesc:cudnnTensorDescriptor_t = ptr::null_mut();
	    let mut zDesc:cudnnTensorDescriptor_t = ptr::null_mut();

	    match cudnnCreateTensorDescriptor(&mut xDesc) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN TensorDescriptor")
	    }
	    match cudnnCreateTensorDescriptor(&mut yDesc) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN TensorDescriptor")
	    }
	    match cudnnCreateTensorDescriptor(&mut zDesc) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to create cuDNN TensorDescriptor")
	    }

	    let batchSize:c_int = 1;
	    let channels:c_int  = 3;
	    let height:c_int = 4;
	    let width:c_int = 4;
	    let dims:[c_int;4] = [batchSize, channels, height, width];
	    let strides:[c_int;4] = [channels*height*width,height*width, width, 1];
	    cudnnSetTensorNdDescriptor(xDesc,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       4,
				       dims.as_ptr(),
				       strides.as_ptr());
	    cudnnSetTensorNdDescriptor(yDesc,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       4,
				       dims.as_ptr(),
				       strides.as_ptr());
	    cudnnSetTensorNdDescriptor(zDesc,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       4,
				       dims.as_ptr(),
				       strides.as_ptr());
	    let dataSize:usize = (batchSize*channels*height*width) as usize;
	    let xDataHost:Vec<c_float> = (0..dataSize).map(|i| i as f32).collect();
	    let yDataHost:Vec<c_float> = (0..dataSize).map(|i| (i*2) as f32).collect();
	    let mut zDataHost:Vec<c_float> = vec![0.0;dataSize];
	    let mut xDataDevice:*mut ::std::os::raw::c_void = ptr::null_mut();
	    let mut yDataDevice:*mut ::std::os::raw::c_void = ptr::null_mut();
	    let mut zDataDevice:*mut ::std::os::raw::c_void = ptr::null_mut();

	    let alloc_size:size_t = (dataSize*std::mem::size_of::<c_float>()).try_into().unwrap();
	    cudaMalloc(&mut xDataDevice,alloc_size);
	    cudaMalloc(&mut yDataDevice,alloc_size);
	    cudaMalloc(&mut zDataDevice,alloc_size);

	    cudaMemcpy(xDataDevice,
		       xDataHost.as_ptr() as *const c_void,
		       alloc_size,
		       cudaMemcpyKind_cudaMemcpyHostToDevice);
	    cudaMemcpy(yDataDevice,
		       yDataHost.as_ptr() as *const c_void,
		       alloc_size,
		       cudaMemcpyKind_cudaMemcpyHostToDevice);
	    let mut opTensorDesc:cudnnOpTensorDescriptor_t = ptr::null_mut();
	    cudnnCreateOpTensorDescriptor(&mut opTensorDesc);
	    cudnnSetOpTensorDescriptor(opTensorDesc,
				       cudnnOpTensorOp_t_CUDNN_OP_TENSOR_ADD,
				       cudnnDataType_t_CUDNN_DATA_FLOAT,
				       cudnnNanPropagation_t_CUDNN_PROPAGATE_NAN);
	    let alpha1:[c_float;1] = [1.0];
	    let alpha2:[c_float;1] = [1.0];
	    let beta:[c_float;1]   = [1.0];

	    cudnnOpTensor(handle, opTensorDesc,
			  alpha1.as_ptr() as *const c_void,
			  xDesc,
			  xDataDevice,
			  alpha2.as_ptr() as *const c_void,
			  yDesc,
			  yDataDevice,
			  beta.as_ptr() as  *const c_void,
			  zDesc,
			  zDataDevice);

	    cudaMemcpy(zDataHost.as_ptr() as *mut c_void,
		       zDataDevice,
		       alloc_size,
		       cudaMemcpyKind_cudaMemcpyDeviceToHost);

	    for (i,v) in zDataHost.iter().enumerate() {
		println!("zDataHost[{i}]={v}");
		assert_eq!(*v, xDataHost[i]+yDataHost[i]);
	    }

	    cudaFree(xDataDevice);
	    cudaFree(yDataDevice);
	    cudaFree(zDataDevice);

	    cudnnDestroyTensorDescriptor(xDesc);
	    cudnnDestroyTensorDescriptor(yDesc);
	    cudnnDestroyTensorDescriptor(zDesc);

	    match cudnnDestroy(handle) {
		cudnnStatus_t_CUDNN_STATUS_SUCCESS => (),
		_ => panic!("Unable to destroy cuDNN context.")
	    }
	}
    }
}

