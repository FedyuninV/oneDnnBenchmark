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
	float* dstBuffer = static_cast<float*>( dst.get_data_handle() );
	for( size_t i = 0; i < bytes / sizeof( float ); ++i ) {
		dstBuffer[i] = src[i];
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
	const float* srcBuffer = static_cast<const float*>( src.get_data_handle() );
	for( size_t i = 0; i < dst.size(); ++i ) {
		dst[i] = srcBuffer[i];
	}
}

static memory::desc no_format( const memory::desc& desc )
{
	return memory::desc( desc.dims(), desc.data_type(), memory::format_tag::any );
}

static memory::desc no_format( const memory& mem )
{
	return no_format( mem.get_desc() );
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

	memory::desc srcMd( no_format( input ) );
	memory::dims weightDim;
	if( isChannelwise ) {
		weightDim = { filter->GetObjectCount() * filter->GetChannelsCount(), 1, 1, filter->GetHeight(), filter->GetWidth() };
	} else {
		weightDim = { filter->GetObjectCount(), filter->GetChannelsCount(), filter->GetHeight(), filter->GetWidth() };
	}
	memory::desc weightMd( weightDim, memory::data_type::f32, memory::format_tag::any );
	memory::desc dstMd( dstDim, memory::data_type::f32, memory::format_tag::any );
	
	// TODO: test with direct instead of auto
	algorithm convAlgo = algorithm::convolution_auto;

	if( conv.IsZeroFreeTerm() ) {
		return convolution_forward::desc( prop_kind::forward_inference, convAlgo, srcMd, weightMd, dstMd,
			{ conv.GetStrideHeight(), conv.GetStrideWidth() }, { conv.GetPaddingHeight(), conv.GetPaddingWidth() },
			{ conv.GetPaddingHeight(), conv.GetPaddingWidth() } );
	} else {
		CPtr<CDnnBlob> freeTerm = conv.GetFreeTermData();
		memory::desc biasMd( { freeTerm->GetDataSize() }, memory::data_type::f32, memory::format_tag::any );
		// TODO: test with direct instead of auto
		return convolution_forward::desc( prop_kind::forward_inference, convAlgo, srcMd, weightMd, biasMd, dstMd,
			{ conv.GetStrideHeight(), conv.GetStrideWidth() }, { conv.GetPaddingHeight(), conv.GetPaddingWidth() },
			{ conv.GetPaddingHeight(), conv.GetPaddingWidth() } );
	}
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

static memory addConv( CDnn& dnn, const CString& convName, const CString& channelwiseOpName, bool addReLU, engine& dnnlEngine, memory& input, memory& toAdd,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
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
		assert( channelwiseOp->IsZeroFreeTerm() );
		assert( conv.GetFilterHeight() == 1 && conv.GetFilterWidth() == 1 );
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
	if( channelwiseOp != nullptr ) {
		dstDims[2] = convOutputSize( static_cast< int >( dstDims[2] ), channelwiseOp->GetFilterHeight(), channelwiseOp->GetStrideHeight(),
			channelwiseOp->GetPaddingHeight(), channelwiseOp->GetDilationHeight() );
		dstDims[3] = convOutputSize( static_cast< int >( dstDims[3] ), channelwiseOp->GetFilterWidth(), channelwiseOp->GetStrideWidth(),
			channelwiseOp->GetPaddingWidth(), channelwiseOp->GetDilationWidth() );
	}

	convolution_forward::desc convDesc = buildConvDesc( conv, input, dnnlEngine, dstDims );

	// Add post-ops if needed
	dnnl::post_ops convPo;
	if( addReLU ) {
		convPo.append_eltwise( 1.f, algorithm::eltwise_relu, 0.f, 0.f );
	}
	if( channelwiseOp != nullptr ) {
		if( channelwiseOp->GetStrideHeight() == 1 ) {
			convPo.append_dw_k3s1p1( memory::data_type::f32, memory::data_type::f32, memory::data_type::f32, 0, {} );
		} else {
			convPo.append_dw_k3s2p1( memory::data_type::f32, memory::data_type::f32, memory::data_type::f32, 0, {} );
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
	memory dstMemory = toAdd == memory() ? memory( convPd.dst_desc(), dnnlEngine )
		: reorderIfNeeded( toAdd, convPd.dst_desc(), fwd, fwdArgs, dnnlEngine );
	
	fwd.push_back( convolution_forward( convPd ) );
	fwdArgs.push_back( { { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_DST, dstMemory } } );
	if( !conv.IsZeroFreeTerm() ) {
		memory biasMemory = reorderIfNeeded( convBias( conv, dnnlEngine ), convPd.bias_desc(), fwd, fwdArgs, dnnlEngine );
		fwdArgs.back().insert( { DNNL_ARG_BIAS, biasMemory } );
	}
	if( channelwiseOp != nullptr ) {
		memory dwPoMemory( { { conv.GetFilterCount(), 1, 1, 3, 3 }, memory::data_type::f32, memory::format_tag::goihw }, dnnlEngine );
		copyToDnnlMemory( *channelwiseOp->GetFilterData(), dwPoMemory );
		fwdArgs.back().insert( { DNNL_ARG_ATTR_POST_OP_DW, dwPoMemory } );
	}
	return dstMemory;
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
	if( conv2->IsZeroFreeTerm() ) {
		// HERE DwConv fused with Conv
		convOutput = addConv( dnn, blockName + "conv1", blockName + "conv2", true, dnnlEngine, input, memory(), fwd, fwdArgs );
	} else {
		convOutput = addConv( dnn, blockName + "conv1", "", true, dnnlEngine, input, memory(), fwd, fwdArgs );
		convOutput = addConv( dnn, blockName + "conv2", "", true, dnnlEngine, convOutput, memory(), fwd, fwdArgs );
	}

	const int firstStride = conv2->GetStrideHeight();
	const CString shortcutName = blockName + "convShortcut";
	memory toSum;
	if( firstStride == 1 && dnn.HasLayer( shortcutName ) ) {
		toSum = addConv( dnn, shortcutName, "", false, dnnlEngine, input, memory(), fwd, fwdArgs );
	} else if( firstStride == 1 ) {
		toSum = input;
	}

	return addConv( dnn, blockName + "conv3", "", false, dnnlEngine, convOutput, toSum, fwd, fwdArgs );
}

static memory addMeanPooling( CDnn& dnn, const CString& poolName, engine& dnnlEngine, memory& input,
	std::vector<primitive>& fwd, std::vector<std::unordered_map<int, memory>>& fwdArgs )
{
	assert( dnn.HasLayer( poolName ) );
	CMeanPoolingLayer& pool = *dynamic_cast<CMeanPoolingLayer*>( dnn.GetLayer( poolName ).Ptr() );

	memory::dims dstDim = input.get_desc().dims();
	dstDim[2] = convOutputSize( static_cast< int >( dstDim[2] ), pool.GetFilterHeight(), pool.GetStrideHeight(), 0, 1 );
	dstDim[3] = convOutputSize( static_cast< int >( dstDim[3] ), pool.GetFilterWidth(), pool.GetStrideWidth(), 0, 1 );
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

	memory::desc srcMd = no_format( input );
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

static void printNetInfo( const CDnn& dnn )
{
	CArray<const char*> layerNames;
	dnn.GetLayerList( layerNames );

	for( int i = 0; i < layerNames.Size(); ++i ) {
		const CBaseLayer* layer = dnn.GetLayer( layerNames[i] );
		if( dynamic_cast<const CSourceLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "Source", layer->GetName() );
		} else if( dynamic_cast<const CSinkLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "Sink", layer->GetName() );
		} else if( dynamic_cast<const CConvLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\t(%s)\n", "Conv", layer->GetName(), dynamic_cast<const CConvLayer*>( layer )->IsZeroFreeTerm() ? "true" : "false" );
		} else if( dynamic_cast<const CChannelwiseConvLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\t(%s)\n", "ChannelwiseConv", layer->GetName(), dynamic_cast<const CChannelwiseConvLayer*>( layer )->IsZeroFreeTerm() ? "true" : "false" );
		} else if( dynamic_cast<const CReLULayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\t%f\n", "ReLU", layer->GetName(), dynamic_cast<const CReLULayer*>( layer )->GetUpperThreshold() );
		} else if( dynamic_cast<const CEltwiseSumLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "Sum", layer->GetName() );
		} else if( dynamic_cast<const CFullyConnectedLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "FC", layer->GetName() );
		} else if( dynamic_cast<const CMaxPoolingLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "MaxPool", layer->GetName() );
		} else if( dynamic_cast<const CMeanPoolingLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "MeanPool", layer->GetName() );
		} else {
			printf( "UNKNOWN:\t%s\n", layer->GetName() );
		}
	}
}

int main( int argc, char** argv )
{
	const CString netName = "MobileNetV2Cifar10";
	const CString inputName = "in";
	const size_t runCount = 10000;
	const bool fusedDepthwise = false;

	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 0, 0 ) );
	engine dnnlEngine( engine::kind::cpu, 0 );
	stream dnnlStream( dnnlEngine );

	CRandom random( 0x54 );
	CDnn dnn( random, *mathEngine );

	loadDnn( dnn, netName );
	if( fusedDepthwise ) {
		CArray<const char*> layerNames;
		dnn.GetLayerList( layerNames );
		for( int i = 0; i < layerNames.Size(); ++i ) {
			CChannelwiseConvLayer* chConv = dynamic_cast<CChannelwiseConvLayer*>( dnn.GetLayer( layerNames[i] ).Ptr() );
			if( chConv != nullptr ) {
				CPtr<CDnnBlob> ft = chConv->GetFreeTermData();
				ft->Clear();
				chConv->SetFreeTermData( ft );
				chConv->SetZeroFreeTerm( true );
			}
		}
	}
	printNetInfo( dnn );
	
	CPtr<CDnnBlob> inputBlob = loadBlob( dnn, netName, inputName );
	memory input( { { inputBlob->GetObjectCount(), inputBlob->GetChannelsCount(), inputBlob->GetHeight(), inputBlob->GetWidth() },
		memory::data_type::f32, memory::format_tag::nchw }, dnnlEngine );
	copyToDnnlMemory( *inputBlob, input );

	std::vector<primitive> fwd;
	std::vector<std::unordered_map<int, memory>> fwdArgs;
	memory output = addConv( dnn, "conv1", "", true, dnnlEngine, input, memory(), fwd, fwdArgs );
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
	output = addConv( dnn, "conv2", "", true, dnnlEngine, output, memory(), fwd, fwdArgs );
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
	in->SetBlob( inputBlob );
	CPtr<CSinkLayer> out = CheckCast<CSinkLayer>( dnn.GetLayer( "out" ) );

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
