#include <NeoML/NeoML.h>
#include <dnnl.hpp>
#include <memory>
#include <iostream>
#include <vector>

using namespace dnnl;

static int convOutputSize( int inputSize, int filterSize, int stride, int padding, int dilation )
{
	assert( dilation == 1 );
	( void ) dilation;
	return ( inputSize + 2 * padding - filterSize ) / stride + 1;
}

static void repackToChannelFirst( const CBlobDesc& desc, const float* hwc, float* chw )
{
	for( int b = 0; b < desc.ObjectCount(); ++b ) {
		for( int hw = 0; hw < desc.Height() * desc.Width(); ++hw ) {
			for( int c = 0; c < desc.Channels(); ++c ) {
				const int hwcIndex = c + desc.Channels() * ( hw + desc.Height() * desc.Width() * b );
				const int chwIndex = hw + desc.Height() * desc.Width() * ( c + desc.Channels() * b );
				chw[chwIndex] = hwc[hwcIndex];
			}
		}
	}
}

static void copyToDnnlMemory( const float* src, memory& dst )
{
	size_t bytes = dst.get_desc().get_size();
	assert( bytes % sizeof( float ) == 0 );
	engine dnnlEngine = dst.get_engine();
	if( dnnlEngine.get_kind() == engine::kind::cpu ) {
		float* dstBuffer = static_cast< float* >( dst.get_data_handle() );
		for( size_t i = 0; i < bytes / sizeof( float ); ++i ) {
			dstBuffer[i] = src[i];
		}
	} else if( dnnlEngine.get_kind() == engine::kind::gpu ) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
		stream dnnlStream( dnnlEngine );
		cl_int ret = clEnqueueWriteBuffer( dnnlStream.get_ocl_command_queue(), dst.get_ocl_mem_object(),
			CL_TRUE, 0, bytes, src, 0, nullptr, nullptr );
		assert( ret == CL_SUCCESS );
#else
		assert( false );
#endif
	} else {
		assert( false );
	}
}

static void copyToDnnlMemory( CDnnBlob& src, memory& dst )
{
	if( src.GetHeight() * src.GetWidth() == 1 || src.GetChannelsCount() == 1 ) {
		float* buffer = src.GetBuffer<float>( 0, src.GetDataSize() );
		copyToDnnlMemory( buffer, dst );
		src.ReleaseBuffer( buffer, false );
	} else {
		float* hwcBuff = src.GetBuffer<float>( 0, src.GetDataSize() );
		std::vector<float> chw( src.GetDataSize() );
		repackToChannelFirst( src.GetDesc(), hwcBuff, chw.data() );
		src.ReleaseBuffer( hwcBuff, false );
		copyToDnnlMemory( chw.data(), dst );
	}
}

static void copyFromDnnlMemory( const memory& src, std::vector<float>& dst )
{
	size_t bytes = src.get_desc().get_size();
	assert( bytes == dst.size() * sizeof( float ) );
	engine dnnlEngine = src.get_engine();
	if( dnnlEngine.get_kind() == engine::kind::cpu ) {
		const float* srcBuffer = static_cast< const float* >( src.get_data_handle() );
		for( size_t i = 0; i < dst.size(); ++i ) {
			dst[i] = srcBuffer[i];
		}
	} else if( dnnlEngine.get_kind() == engine::kind::gpu ) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
		stream dnnlStream( dnnlEngine );
		cl_int ret = clEnqueueReadBuffer( dnnlStream.get_ocl_command_queue(), src.get_ocl_mem_object(),
			CL_TRUE, 0, bytes, dst.data(), 0, nullptr, nullptr );
		assert( ret == CL_SUCCESS );
#else
		assert( false );
#endif
	} else {
		assert( false );
	}
}

static memory::desc noFormat( const memory::desc& desc )
{
	return memory::desc( desc.dims(), desc.data_type(), memory::format_tag::any );
}

static memory::desc noFormat( const memory& mem )
{
	return noFormat( mem.get_desc() );
}

static memory reorderIfNeeded( const memory& src, const memory::desc& dstDesc, std::vector<primitive>& fwd,
	std::vector<std::unordered_map<int, memory>>& fwdArgs, engine& dnnlEngine )
{
	if( src.get_desc() == dstDesc ) {
		return src;
	}

	memory result( dstDesc, dnnlEngine );
	fwd.push_back( reorder( src, result ) );
	fwdArgs.push_back( { { DNNL_ARG_FROM, src }, { DNNL_ARG_TO, result } } );
	return result;
}

static convolution_forward::desc buildConvDesc( const CBaseConvLayer& conv, const memory& input, engine& dnnlEngine, memory::dims dstDim )
{
	CPtr<CDnnBlob> filter = conv.GetFilterData();
	const bool isChannelwise = dynamic_cast<const CChannelwiseConvLayer*>( &conv ) != nullptr;

	memory::desc srcMd( noFormat( input ) );
	memory::dims weightDim;
	if( isChannelwise ) {
		weightDim = { static_cast<memory::dim>( filter->GetObjectCount() ) * filter->GetChannelsCount(), 1, 1, filter->GetHeight(), filter->GetWidth() };
	} else {
		weightDim = { filter->GetObjectCount(), filter->GetChannelsCount(), filter->GetHeight(), filter->GetWidth() };
	}
	memory::desc weightMd( weightDim, memory::data_type::f32, memory::format_tag::any );
	memory::desc biasMd;
	memory::desc dstMd( dstDim, memory::data_type::f32, memory::format_tag::any );
	
	if( !conv.IsZeroFreeTerm() ) {
		biasMd = memory::desc( { conv.GetFilterCount() }, memory::data_type::f32, memory::format_tag::any );
	}

	// TODO: test with direct instead of auto
	return convolution_forward::desc( prop_kind::forward_inference, algorithm::convolution_auto,
		srcMd, weightMd, biasMd, dstMd,
		{ conv.GetStrideHeight(), conv.GetStrideWidth() }, { conv.GetPaddingHeight(), conv.GetPaddingWidth() },
		{ conv.GetPaddingHeight(), conv.GetPaddingWidth() } );
}

static memory convFilter( const CBaseConvLayer& conv, engine& dnnlEngine )
{
	CPtr<CDnnBlob> filter = conv.GetFilterData();
	assert( filter != nullptr );
	const bool isChannelwise = dynamic_cast<const CChannelwiseConvLayer*>( &conv ) != nullptr;

	memory::dims filterDim;
	memory::format_tag filterFormat;
	if( isChannelwise ) {
		filterDim = { filter->GetObjectCount() * filter->GetChannelsCount(), 1, 1, filter->GetHeight(), filter->GetWidth() };
		filterFormat = memory::format_tag::goihw;
	} else {
		filterDim = { filter->GetObjectCount(), filter->GetChannelsCount(), filter->GetHeight(), filter->GetWidth() };
		filterFormat = memory::format_tag::oihw;
	}
	memory::desc filterMd( filterDim, memory::data_type::f32, filterFormat );
	memory filterMemory( filterMd, dnnlEngine );
	copyToDnnlMemory( *filter, filterMemory );
	return filterMemory;
}

static memory convBias( const CBaseConvLayer& conv, engine& dnnlEngine )
{
	CPtr<CDnnBlob> freeTerm = conv.GetFreeTermData();
	assert( freeTerm != nullptr );

	memory::desc biasMd( { freeTerm->GetDataSize() }, memory::data_type::f32, memory::format_tag::x );
	memory biasMemory( biasMd, dnnlEngine );
	copyToDnnlMemory( *freeTerm, biasMemory );
	return biasMemory;
}

static memory addConv( CDnn& dnn, const CString& convName, const CString& channelwiseOpName, bool addReLU, engine& dnnlEngine,
	memory& input, memory& toAdd, std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	assert( dnn.HasLayer( convName ) );
	CBaseConvLayer& conv = *dynamic_cast<CBaseConvLayer*>( dnn.GetLayer( convName ).Ptr() );
	const bool isChannelwise = dynamic_cast<CChannelwiseConvLayer*>( &conv ) != nullptr;

	CChannelwiseConvLayer* channelwiseOp = nullptr;
	if( channelwiseOpName != "" ) {
		assert( dnn.HasLayer( channelwiseOpName ) );
		assert( !isChannelwise );
		channelwiseOp = dynamic_cast<CChannelwiseConvLayer*>( dnn.GetLayer( channelwiseOpName ).Ptr() );
		assert( channelwiseOp != nullptr );
		assert( channelwiseOp->GetFilterHeight() == 3 && channelwiseOp->GetFilterWidth() == 3 );
		assert( channelwiseOp->GetStrideHeight() == channelwiseOp->GetStrideWidth()
			&& ( channelwiseOp->GetStrideHeight() == 1 || channelwiseOp->GetStrideHeight() == 2 ) );
		assert( channelwiseOp->GetPaddingHeight() == 1 && channelwiseOp->GetPaddingWidth() == 1 );
		assert( conv.GetFilterHeight() == 1 && conv.GetFilterWidth() == 1 );
		assert( !channelwiseOp->IsZeroFreeTerm() );
	}

	memory::dims dstDims( input.get_desc().dims() );
	assert( dstDims.size() == 4 );
	if( !isChannelwise ) {
		dstDims[1] = conv.GetFilterCount();
	}
	dstDims[2] = convOutputSize( static_cast<int>( dstDims[2] ), conv.GetFilterHeight(), conv.GetStrideHeight(),
		conv.GetPaddingHeight(), conv.GetDilationHeight() );
	dstDims[3] = convOutputSize( static_cast<int>( dstDims[3] ), conv.GetFilterWidth(), conv.GetStrideWidth(),
		conv.GetPaddingWidth(), conv.GetDilationWidth() );

	convolution_forward::desc convDesc = buildConvDesc( conv, input, dnnlEngine, dstDims );

	// Add post-ops if needed
	dnnl::post_ops convPo;
	if( addReLU ) {
		convPo.append_eltwise( 1.f, algorithm::eltwise_relu, 0.f, 0.f );
	}
	if( channelwiseOp != nullptr ) {
		if( channelwiseOp->GetStrideHeight() == 1 ) {
			convPo.append_dw_k3s1p1( memory::data_type::f32,
				channelwiseOp->IsZeroFreeTerm() ? memory::data_type::undef : memory::data_type::f32,
				memory::data_type::f32, 0, { 1.f } );
		} else {
			convPo.append_dw_k3s2p1( memory::data_type::f32,
				channelwiseOp->IsZeroFreeTerm() ? memory::data_type::undef : memory::data_type::f32,
				memory::data_type::f32, 0, { 1.f } );
		}
		if( addReLU ) {
			convPo.append_eltwise( 1.f, algorithm::eltwise_relu, 0.f, 0.f );
		}
	}
	if( toAdd != memory() ) {
		convPo.append_sum();
	}

	primitive_attr convPa;
	convPa.set_post_ops( convPo );
	convolution_forward::primitive_desc convPd( convDesc, convPa, dnnlEngine );

	memory srcMemory = reorderIfNeeded( input, convPd.src_desc(), fwd, fwdArgs, dnnlEngine );
	memory weightMemory = reorderIfNeeded( convFilter( conv, dnnlEngine ), convPd.weights_desc(), fwd, fwdArgs, dnnlEngine );
	memory biasMemory;
	memory dstMemory = toAdd == memory() ? memory( convPd.dst_desc(), dnnlEngine )
		: reorderIfNeeded( toAdd, convPd.dst_desc(), fwd, fwdArgs, dnnlEngine );
	if( !conv.IsZeroFreeTerm() ) {
		biasMemory = reorderIfNeeded( convBias( conv, dnnlEngine ), convPd.bias_desc(), fwd, fwdArgs, dnnlEngine );
	}

	memory dwWeightMemory;
	memory dwBiasMemory;
	if( channelwiseOp != nullptr ) {
		memory::desc dwWeightMd = convPd.query_md( query::exec_arg_md, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS );
		dwWeightMemory = reorderIfNeeded( convFilter( *channelwiseOp, dnnlEngine ), dwWeightMd, fwd, fwdArgs, dnnlEngine );
		if( !channelwiseOp->IsZeroFreeTerm() ) {
			memory::desc dwBiasMd = convPd.query_md( query::exec_arg_md, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS );
			dwBiasMemory = reorderIfNeeded( convBias( *channelwiseOp, dnnlEngine ), dwBiasMd, fwd, fwdArgs, dnnlEngine );
		}
	}
	
	fwd.push_back( convolution_forward( convPd ) );
	fwdArgs.push_back( { { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_DST, dstMemory },
		{ DNNL_ARG_BIAS, biasMemory }, { DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, dwWeightMemory },
		{ DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS, dwBiasMemory } } );

	return dstMemory;
}

static memory addConv( CDnn& dnn, const CString& convName, const CString& channelwiseOpName, bool addReLU, engine& dnnlEngine,
	memory& input, std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	return addConv( dnn, convName, channelwiseOpName, addReLU, dnnlEngine, input, memory(), fwd, fwdArgs );
}

static memory addConv( CDnn& dnn, const CString& convName, bool addReLU, engine& dnnlEngine, memory& input, memory& toAdd,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	return addConv( dnn, convName, "", addReLU, dnnlEngine, input, toAdd, fwd, fwdArgs );
}

static memory addConv( CDnn& dnn, const CString& convName, bool addReLU, engine& dnnlEngine, memory& input,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	return addConv( dnn, convName, "", addReLU, dnnlEngine, input, memory(), fwd, fwdArgs );
}

static memory addBlock( CDnn& dnn, const CString& blockName, engine& dnnlEngine, memory& input,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	assert( dnn.HasLayer( blockName + "conv1" ) );
	assert( dnn.HasLayer( blockName + "conv2" ) );
	assert( dnn.HasLayer( blockName + "conv3" ) );

	CPtr<CConvLayer> conv1 = CheckCast<CConvLayer>( dnn.GetLayer( blockName + "conv1" ) );
	CPtr<CChannelwiseConvLayer> conv2 = CheckCast<CChannelwiseConvLayer>( dnn.GetLayer( blockName + "conv2" ) );
	CPtr<CConvLayer> conv3 = CheckCast<CConvLayer>( dnn.GetLayer( blockName + "conv3" ) );

	memory convOutput;
	if( dnnlEngine.get_kind() == engine::kind::cpu
		&& conv2->GetStrideHeight() == conv2->GetStrideWidth()
		&& ( conv2->GetStrideWidth() == 1 || conv2->GetStrideWidth() == 2 )
		&& conv2->GetPaddingHeight() == 1 && conv2->GetPaddingWidth() == 1
		&& conv2->GetDilationHeight() == 1 && conv2->GetDilationWidth() == 1
		&& conv2->GetFilterHeight() == 3 && conv2->GetFilterWidth() == 3 )
	{
		// Here DwConv is fused with Conv (after fixes)
		convOutput = addConv( dnn, blockName + "conv1", blockName + "conv2", true, dnnlEngine, input, fwd, fwdArgs );
	} else {
		convOutput = addConv( dnn, blockName + "conv1", true, dnnlEngine, input, fwd, fwdArgs );
		convOutput = addConv( dnn, blockName + "conv2", true, dnnlEngine, convOutput, fwd, fwdArgs );
	}

	const int firstStride = conv2->GetStrideHeight();
	const CString shortcutName = blockName + "convShortcut";
	memory toSum;
	if( firstStride == 1 && dnn.HasLayer( shortcutName ) ) {
		toSum = addConv( dnn, shortcutName, "", false, dnnlEngine, input, memory(), fwd, fwdArgs );
	} else if( firstStride == 1 ) {
		toSum = input;
	}

	return addConv( dnn, blockName + "conv3", false, dnnlEngine, convOutput, toSum, fwd, fwdArgs );
}

static memory addMeanPooling( CDnn& dnn, const CString& poolName, engine& dnnlEngine, memory& input,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	assert( dnn.HasLayer( poolName ) );
	CMeanPoolingLayer& pool = *dynamic_cast<CMeanPoolingLayer*>( dnn.GetLayer( poolName ).Ptr() );

	memory::dims dstDim = input.get_desc().dims();
	dstDim[2] = convOutputSize( static_cast<int>( dstDim[2] ), pool.GetFilterHeight(), pool.GetStrideHeight(), 0, 1 );
	dstDim[3] = convOutputSize( static_cast<int>( dstDim[3] ), pool.GetFilterWidth(), pool.GetStrideWidth(), 0, 1 );
	memory::desc dstMd( dstDim, memory::data_type::f32, memory::format_tag::any );

	pooling_forward::desc poolDesc( prop_kind::forward_inference, algorithm::pooling_avg, input.get_desc(), dstMd,
		{ pool.GetStrideHeight(), pool.GetStrideWidth() }, { pool.GetFilterHeight(), pool.GetFilterWidth() },
		{ 0, 0 }, { 0, 0 } );

	pooling_forward::primitive_desc poolPd( poolDesc, dnnlEngine );

	memory dstMemory( poolPd.dst_desc(), dnnlEngine );
	fwd.push_back( pooling_forward( poolPd ) );
	fwdArgs.push_back( { { DNNL_ARG_SRC, input }, { DNNL_ARG_DST, dstMemory } } );
	return dstMemory;
}

static memory addFc( CDnn& dnn, const CString& fcName, engine& dnnlEngine, memory& input,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	assert( dnn.HasLayer( fcName ) );
	CFullyConnectedLayer& fc = *dynamic_cast<CFullyConnectedLayer*>( dnn.GetLayer( fcName ).Ptr() );
	memory::dims srcDim = input.get_desc().dims();

	memory::desc srcMd = noFormat( input );
	memory::desc weightMd( { fc.GetNumberOfElements(), srcDim[1], 1, 1 }, memory::data_type::f32, memory::format_tag::any );
	memory::desc dstMd( { srcDim[0], fc.GetNumberOfElements() }, memory::data_type::f32, memory::format_tag::any );

	memory weightMemory( { { fc.GetNumberOfElements(), srcDim[1], 1, 1 }, memory::data_type::f32, memory::format_tag::nchw }, dnnlEngine );
	copyToDnnlMemory( *fc.GetWeightsData(), weightMemory );

	if( fc.IsZeroFreeTerm() ) {
		inner_product_forward::primitive_desc ipPd( { prop_kind::forward_inference, srcMd, weightMd, dstMd }, dnnlEngine );
		memory srcMemory = reorderIfNeeded( input, ipPd.src_desc(), fwd, fwdArgs, dnnlEngine );
		weightMemory = reorderIfNeeded( weightMemory, ipPd.weights_desc(), fwd, fwdArgs, dnnlEngine );
		memory dstMemory( ipPd.dst_desc(), dnnlEngine );
		fwd.push_back( inner_product_forward( ipPd ) );
		fwdArgs.push_back( { { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_DST, dstMemory } } );
		return dstMemory;
	}

	memory::desc biasMd( { fc.GetNumberOfElements() }, memory::data_type::f32, memory::format_tag::any );
	memory biasMemory( { { fc.GetNumberOfElements() }, memory::data_type::f32, memory::format_tag::x }, dnnlEngine );
	copyToDnnlMemory( *fc.GetFreeTermData(), biasMemory );

	inner_product_forward::primitive_desc ipPd( { prop_kind::forward_inference, srcMd, weightMd, biasMd, dstMd }, dnnlEngine );
	memory srcMemory = reorderIfNeeded( input, ipPd.src_desc(), fwd, fwdArgs, dnnlEngine );
	weightMemory = reorderIfNeeded( weightMemory, ipPd.weights_desc(), fwd, fwdArgs, dnnlEngine );
	biasMemory = reorderIfNeeded( biasMemory, ipPd.bias_desc(), fwd, fwdArgs, dnnlEngine );
	memory dstMemory( ipPd.dst_desc(), dnnlEngine );
	fwd.push_back( inner_product_forward( ipPd ) );
	fwdArgs.push_back( { { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_BIAS, biasMemory }, { DNNL_ARG_DST, dstMemory } } );
	return dstMemory;
}

static memory convertOutput( memory& currOutput, std::vector<primitive>& fwd,
	std::vector<std::unordered_map<int, memory>>& fwdArgs, engine& dnnlEngine )
{
	memory::dims outDims( currOutput.get_desc().dims() );
	memory::format_tag preferredFormat;
	switch( outDims.size() ) {
	case 1:
		preferredFormat = memory::format_tag::x;
		break;
	case 2:
		preferredFormat = memory::format_tag::nc;
		break;
	case 4:
		preferredFormat = memory::format_tag::nchw;
		break;
	default:
		assert( false );
	}
	memory::desc preferredOutputDesc( outDims, memory::data_type::f32, preferredFormat );
	return reorderIfNeeded( currOutput, preferredOutputDesc, fwd, fwdArgs, dnnlEngine );
}

static void loadDnn( CDnn& dnn, const CString& netName )
{
	CArchiveFile file( netName + ".cnnarch", CArchive::load );
	CArchive archive( &file, CArchive::load );
	archive.Serialize( dnn );
	archive.Close();
	file.Close();
}

CPtr<CDnnBlob> loadBlob( CDnn& dnn, const CString& netName, const CString& inputName )
{
	CPtr<CDnnBlob> inputBlob = new CDnnBlob( dnn.GetMathEngine() );
	CArchiveFile file( netName + "." + inputName + ".input", CArchive::load );
	CArchive archive( &file, CArchive::load );
	inputBlob->Serialize( archive );
	archive.Close();
	file.Close();
	CheckCast<CSourceLayer>( dnn.GetLayer( inputName ) )->SetBlob( inputBlob );
	return inputBlob;
}

int main( int argc, char** argv )
{
	const CString netName = "MobileNetV2Cifar10";
	const CString inputName = "in";
	const size_t runCount = 100;
	const int imageSizeMultiplier = 16;
	engine::kind engineKind = engine::kind::gpu;

	std::unique_ptr<IMathEngine> mathEngine;
	if( engineKind == engine::kind::cpu ) {
		mathEngine.reset( CreateCpuMathEngine( 0, 0 ) );
	} else {
		std::unique_ptr<IGpuMathEngineManager> manager( CreateGpuMathEngineManager() );
		const int gpuMathEngineCount = manager->GetMathEngineCount();
		for( int i = 0; i < gpuMathEngineCount; ++i ) {
			CMathEngineInfo mathEngineInfo;
			manager->GetMathEngineInfo( i, mathEngineInfo );
			if( mathEngineInfo.Type == MET_Vulkan && std::string( mathEngineInfo.Name ).substr( 0, 5 ) == "Intel" ) {
				std::cout << "GPU:\t" << mathEngineInfo.Name << std::endl;
				mathEngine.reset( manager->CreateMathEngine( i, 0 ) );
				break;
			}
		}
	}

	if( mathEngine == nullptr ) {
		std::cerr << "Failed to create mathEngine" << std::endl;
		return 1;
	}

	engine dnnlEngine( engineKind, 0 );
	stream dnnlStream( dnnlEngine );

	CRandom random( 0x54 );
	CDnn dnn( random, *mathEngine );

	loadDnn( dnn, netName );

	CPtr<CDnnBlob> inputBlob = loadBlob( dnn, netName, inputName );
	if( imageSizeMultiplier != 1 ) {
		inputBlob = CDnnBlob::Create2DImageBlob( *mathEngine, CT_Float, 1, 1,
			32 * imageSizeMultiplier, 32 * imageSizeMultiplier, 3 );
		float* buff = inputBlob->GetBuffer<float>( 0, inputBlob->GetDataSize() );
		for( int i = 0; i < inputBlob->GetDataSize(); ++i ) {
			buff[i] = static_cast<float>( random.Uniform( -1, 2 ) );
		}
		inputBlob->ReleaseBuffer( buff, true );
		CPtr<CMeanPoolingLayer> pool = CheckCast<CMeanPoolingLayer>( dnn.GetLayer( "pool" ) );
		pool->SetFilterHeight( pool->GetFilterHeight() * imageSizeMultiplier );
		pool->SetFilterWidth( pool->GetFilterWidth() * imageSizeMultiplier );
	}
	memory input( { { inputBlob->GetObjectCount(), inputBlob->GetChannelsCount(), inputBlob->GetHeight(), inputBlob->GetWidth() },
		memory::data_type::f32, memory::format_tag::nchw }, dnnlEngine );
	copyToDnnlMemory( *inputBlob, input );

	std::vector<primitive> fwd;
	std::vector<std::unordered_map<int, memory>> fwdArgs;
	memory output = addConv( dnn, "conv1", true, dnnlEngine, input, fwd, fwdArgs );
	output = addBlock( dnn, "block0", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block10", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block11", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block20", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block21", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block22", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block30", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block31", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block32", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block33", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block40", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block41", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block42", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block50", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block51", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block52", dnnlEngine, output, fwd, fwdArgs );
	output = addBlock( dnn, "block6", dnnlEngine, output, fwd, fwdArgs );
	output = addConv( dnn, "conv2", true, dnnlEngine, output, fwd, fwdArgs );
	output = addMeanPooling( dnn, "pool", dnnlEngine, output, fwd, fwdArgs );
	output = addFc( dnn, "fc", dnnlEngine, output, fwd, fwdArgs );
	output = convertOutput( output, fwd, fwdArgs, dnnlEngine );

	const size_t outputBytes = output.get_desc().get_size();
	assert( outputBytes % sizeof( float ) == 0 );
	std::vector<float> actual( outputBytes / sizeof( float ) );

	for( size_t i = 0; i < fwd.size(); ++i ) {
		fwd.at( i ).execute( dnnlStream, fwdArgs.at( i ) );
	}
	dnnlStream.wait();
	copyFromDnnlMemory( output, actual );

	{
		auto counters = dnn.GetMathEngine().CreatePerformanceCounters();
		counters->Synchronise();

		for( size_t run = 1; run <= runCount; ++run ) {
			for( size_t i = 0; i < fwd.size(); ++i ) {
				fwd.at( i ).execute( dnnlStream, fwdArgs.at( i ) );
			}
			dnnlStream.wait();
			copyFromDnnlMemory( output, actual );
		}

		counters->Synchronise();
		std::cout << "*** ONE DNN ***" << std::endl;
		for( const auto& counter : *counters ) {
			std::cout << counter.Name << ": " << counter.Value << std::endl;
		}
		std::cout << std::endl;
	}

	CPtr<CSourceLayer> in = CheckCast<CSourceLayer>( dnn.GetLayer( "in" ) );
	CPtr<CSinkLayer> out = CheckCast<CSinkLayer>( dnn.GetLayer( "out" ) );

	in->SetBlob( inputBlob );
	dnn.RunOnce();
	const CPtr<CDnnBlob>& outputBlob = out->GetBlob();
	std::vector<float> expected( outputBlob->GetDataSize() );
	float* buffer = outputBlob->GetBuffer<float>( 0, outputBlob->GetDataSize() );
	repackToChannelFirst( outputBlob->GetDesc(), buffer, expected.data() );
	outputBlob->ReleaseBuffer( buffer, false );

	std::vector<float> buff( outputBlob->GetDataSize() );
	{
		auto counters = dnn.GetMathEngine().CreatePerformanceCounters();
		counters->Synchronise();

		for( size_t run = 1; run <= runCount; ++run ) {
			dnn.RunOnce();
			out->GetBlob()->CopyTo( buff.data() );
		}

		counters->Synchronise();
		std::cout << "***  NeoML  ***" << std::endl;
		for( const auto& counter : *counters ) {
			std::cout << counter.Name << ": " << counter.Value << std::endl;
		}
		std::cout << std::endl;
	}

	float maxAbsErr = 0;
	assert( expected.size() == actual.size() );
	for( size_t i = 0; i < expected.size(); ++i ) {
		maxAbsErr = max( maxAbsErr, fabsf( expected[i] - actual[i] ) );
	}

	std::cout << "Max abs err: " << maxAbsErr << std::endl;

	return 0;
}
