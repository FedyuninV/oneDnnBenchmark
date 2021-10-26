#include <NeoML/NeoML.h>
#include <dnnl.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace dnnl;

typedef std::unordered_map<int, memory> primitiveArg;
typedef IPerformanceCounters::CCounter::TCounterType TTimerType;

class CDnnlNet {
public:
	explicit CDnnlNet( engine::kind kind );

	const engine& Engine() const { return dnnlEngine; }
	engine& Engine() { return dnnlEngine; }

	primitive& LastPrimitive() { assert( !primitives.empty() ); return primitives.back(); }
	primitiveArg& LastArg() { assert( !args.empty() ); return args.back(); }

	void AddPrimitive( const primitive& p, const primitiveArg& arg, const std::string& name )
		{ primitives.push_back( p ); args.push_back( arg ); names.push_back( name ); }
	void AddPrimitive( primitive&& p, primitiveArg&& arg, std::string&& name )
		{ primitives.push_back( p ); args.push_back( arg ); names.push_back( name ); }

	void Execute();
	void Profile( IMathEngine& mathEngine );

	const std::vector<std::string>& PrimitiveNames() const { return names; }
	TTimerType PrimitiveTime( const std::string& name )
		{ assert( times.find( name ) != times.end() ); return times[name]; }

private:
	engine dnnlEngine;
	stream dnnlStream;
	std::vector<primitive> primitives;
	std::vector<primitiveArg> args;
	std::vector<std::string> names;
	std::unordered_map<std::string, TTimerType> times;
};

CDnnlNet::CDnnlNet( engine::kind kind ) :
	dnnlEngine( kind, 0 ),
	dnnlStream( dnnlEngine )
{
}

void CDnnlNet::Execute()
{
	assert( primitives.size() == args.size() );
	for( size_t i = 0; i < primitives.size(); ++i ) {
		primitives[i].execute( dnnlStream, args[i] );
	}
	dnnlStream.wait();
}

void CDnnlNet::Profile( IMathEngine& mathEngine )
{
	assert( primitives.size() == args.size() );
	assert( dnnlEngine.get_kind() == engine::kind::cpu );

	std::unique_ptr<IPerformanceCounters> counters( mathEngine.CreatePerformanceCounters() );
	for( size_t i = 0; i < primitives.size(); ++i ) {
		counters->Synchronise();
		primitives[i].execute( dnnlStream, args[i] );
		dnnlStream.wait();
		counters->Synchronise();
		times[names[i]] += ( *counters )[0].Value;
	}
}

static int convOutputSize( int inputSize, int filterSize, int stride, int padding, int dilation )
{
	return ( inputSize + 2 * padding - ( ( filterSize - 1 ) * dilation + 1 ) ) / stride + 1;
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

static void copyToDnnlMemory( const CDnnBlob& _src, memory& dst )
{
	CDnnBlob& src = const_cast<CDnnBlob&>( _src );
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

static memory reorderIfNeeded( const memory& src, const memory::desc& dstDesc, const std::string& baseName, CDnnlNet& net )
{
	if( src.get_desc() == dstDesc ) {
		return src;
	}

	memory result( dstDesc, net.Engine() );
	net.AddPrimitive( reorder( src, result ), { { DNNL_ARG_FROM, src }, { DNNL_ARG_TO, result } }, baseName + "Reorder" );
	return result;
}

static convolution_forward::desc buildConvDesc( const CBaseConvLayer& conv, const memory& input, engine& dnnlEngine, memory::dims dstDim )
{
	CPtr<CDnnBlob> filter = conv.GetFilterData();
	const bool isChannelwise = dynamic_cast< const CChannelwiseConvLayer* >( &conv ) != nullptr;

	memory::desc srcMd( noFormat( input ) );
	memory::dims weightDim;
	if( isChannelwise ) {
		weightDim = { static_cast< memory::dim >( filter->GetObjectCount() ) * filter->GetChannelsCount(), 1, 1, filter->GetHeight(), filter->GetWidth() };
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
		srcMd, weightMd, biasMd, dstMd, { conv.GetStrideHeight(), conv.GetStrideWidth() },
		{ conv.GetDilationHeight() - 1, conv.GetDilationWidth() - 1 }, { conv.GetPaddingHeight(), conv.GetPaddingWidth() },
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

static memory addConv( const CDnn& dnn, const CString& convName, const CString& channelwiseOpName, bool addReLU,
	memory& input, memory& toAdd, CDnnlNet& net )
{
	assert( dnn.HasLayer( convName ) );
	const CBaseConvLayer& conv = *dynamic_cast<const CBaseConvLayer*>( dnn.GetLayer( convName ).Ptr() );
	const bool isChannelwise = dynamic_cast<const CChannelwiseConvLayer*>( &conv ) != nullptr;

	const CChannelwiseConvLayer* channelwiseOp = nullptr;
	if( channelwiseOpName != "" ) {
		assert( dnn.HasLayer( channelwiseOpName ) );
		assert( !isChannelwise );
		channelwiseOp = dynamic_cast<const CChannelwiseConvLayer*>( dnn.GetLayer( channelwiseOpName ).Ptr() );
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

	convolution_forward::desc convDesc = buildConvDesc( conv, input, net.Engine(), dstDims );
	std::string primitiveName = convName;

	// Add post-ops if needed
	dnnl::post_ops convPo;
	if( addReLU ) {
		convPo.append_eltwise( 1.f, algorithm::eltwise_relu, 0.f, 0.f );
		primitiveName += "Relu";
	}
	if( channelwiseOp != nullptr ) {
		if( channelwiseOp->GetStrideHeight() == 1 ) {
			primitiveName += "Dws1";
			convPo.append_dw_k3s1p1( memory::data_type::f32,
				channelwiseOp->IsZeroFreeTerm() ? memory::data_type::undef : memory::data_type::f32,
				memory::data_type::f32, 0, { 1.f } );
		} else {
			primitiveName += "Dws2";
			convPo.append_dw_k3s2p1( memory::data_type::f32,
				channelwiseOp->IsZeroFreeTerm() ? memory::data_type::undef : memory::data_type::f32,
				memory::data_type::f32, 0, { 1.f } );
		}
		if( addReLU ) {
			convPo.append_eltwise( 1.f, algorithm::eltwise_relu, 0.f, 0.f );
			primitiveName += "Relu";
		}
	}
	if( toAdd != memory() ) {
		convPo.append_sum();
		primitiveName += "Sum";
	}

	primitive_attr convPa;
	convPa.set_post_ops( convPo );
	convolution_forward::primitive_desc convPd( convDesc, convPa, net.Engine() );

	memory srcMemory = reorderIfNeeded( input, convPd.src_desc(), convName + "Src", net );
	memory weightMemory = reorderIfNeeded( convFilter( conv, net.Engine() ), convPd.weights_desc(), convName + "Weight", net );
	memory biasMemory;
	memory dstMemory = toAdd == memory() ? memory( convPd.dst_desc(), net.Engine() )
		: reorderIfNeeded( toAdd, convPd.dst_desc(), convName + "ToAdd", net );
	if( !conv.IsZeroFreeTerm() ) {
		biasMemory = reorderIfNeeded( convBias( conv, net.Engine() ), convPd.bias_desc(), convName + "Bias", net );
	}

	memory dwWeightMemory;
	memory dwBiasMemory;
	if( channelwiseOp != nullptr ) {
		memory::desc dwWeightMd = convPd.query_md( query::exec_arg_md, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS );
		dwWeightMemory = reorderIfNeeded( convFilter( *channelwiseOp, net.Engine() ), dwWeightMd, convName + "DwWeight", net );
		if( !channelwiseOp->IsZeroFreeTerm() ) {
			memory::desc dwBiasMd = convPd.query_md( query::exec_arg_md, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS );
			dwBiasMemory = reorderIfNeeded( convBias( *channelwiseOp, net.Engine() ), dwBiasMd, convName + "DwBias", net );
		}
	}
	
	net.AddPrimitive( convolution_forward( convPd ),
		{ { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_DST, dstMemory },
			{ DNNL_ARG_BIAS, biasMemory }, { DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, dwWeightMemory },
			{ DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS, dwBiasMemory } }, primitiveName );

	return dstMemory;
}

static memory addConv( const CDnn& dnn, const CString& convName, const CString& channelwiseOpName, bool addReLU,
	memory& input, CDnnlNet& net )
{
	return addConv( dnn, convName, channelwiseOpName, addReLU, input, memory(), net );
}

static memory addConv( const CDnn& dnn, const CString& convName, bool addReLU, memory& input, memory& toAdd, CDnnlNet& net )
{
	return addConv( dnn, convName, "", addReLU, input, toAdd, net );
}

static memory addConv( const CDnn& dnn, const CString& convName, bool addReLU, memory& input, CDnnlNet& net )
{
	return addConv( dnn, convName, "", addReLU, input, memory(), net );
}

static memory addBlock( const CDnn& dnn, const CString& blockName, memory& input, CDnnlNet& net )
{
	assert( dnn.HasLayer( blockName + "conv1" ) );
	assert( dnn.HasLayer( blockName + "conv2" ) );
	assert( dnn.HasLayer( blockName + "conv3" ) );

	CPtr<const CConvLayer> conv1 = CheckCast<const CConvLayer>( dnn.GetLayer( blockName + "conv1" ) );
	CPtr<const CChannelwiseConvLayer> conv2 = CheckCast<const CChannelwiseConvLayer>( dnn.GetLayer( blockName + "conv2" ) );
	CPtr<const CConvLayer> conv3 = CheckCast<const CConvLayer>( dnn.GetLayer( blockName + "conv3" ) );

	memory convOutput;
	if( net.Engine().get_kind() == engine::kind::cpu
		&& conv2->GetStrideHeight() == conv2->GetStrideWidth()
		&& ( conv2->GetStrideWidth() == 1 || conv2->GetStrideWidth() == 2 )
		&& conv2->GetPaddingHeight() == 1 && conv2->GetPaddingWidth() == 1
		&& conv2->GetDilationHeight() == 1 && conv2->GetDilationWidth() == 1
		&& conv2->GetFilterHeight() == 3 && conv2->GetFilterWidth() == 3
		&& conv1->GetFilterHeight() == 1 && conv1->GetFilterWidth() == 1 )
	{
		// Here DwConv is fused with Conv (after fixes)
		convOutput = addConv( dnn, blockName + "conv1", blockName + "conv2", true, input, net );
	} else {
		convOutput = addConv( dnn, blockName + "conv1", true, input, net );
		convOutput = addConv( dnn, blockName + "conv2", true, convOutput, net );
	}

	const int firstStride = conv2->GetStrideHeight();
	const CString shortcutName = blockName + "convShortcut";
	memory toSum;
	if( firstStride == 1 && dnn.HasLayer( shortcutName ) ) {
		toSum = addConv( dnn, shortcutName, "", false, input, memory(), net );
	} else if( firstStride == 1 ) {
		toSum = input;
	}

	return addConv( dnn, blockName + "conv3", false, convOutput, toSum, net );
}

static memory addMeanPooling( const CDnn& dnn, const CString& poolName, memory& input, CDnnlNet& net )
{
	assert( dnn.HasLayer( poolName ) );
	const CMeanPoolingLayer& pool = *dynamic_cast<const CMeanPoolingLayer*>( dnn.GetLayer( poolName ).Ptr() );

	memory::dims dstDim = input.get_desc().dims();
	dstDim[2] = convOutputSize( static_cast<int>( dstDim[2] ), pool.GetFilterHeight(), pool.GetStrideHeight(), 0, 1 );
	dstDim[3] = convOutputSize( static_cast<int>( dstDim[3] ), pool.GetFilterWidth(), pool.GetStrideWidth(), 0, 1 );
	memory::desc dstMd( dstDim, memory::data_type::f32, memory::format_tag::any );

	pooling_forward::desc poolDesc( prop_kind::forward_inference, algorithm::pooling_avg, input.get_desc(), dstMd,
		{ pool.GetStrideHeight(), pool.GetStrideWidth() }, { pool.GetFilterHeight(), pool.GetFilterWidth() },
		{ 0, 0 }, { 0, 0 } );

	pooling_forward::primitive_desc poolPd( poolDesc, net.Engine() );

	memory dstMemory( poolPd.dst_desc(), net.Engine() );
	net.AddPrimitive( pooling_forward( poolPd ), { { DNNL_ARG_SRC, input }, { DNNL_ARG_DST, dstMemory } }, poolName );
	return dstMemory;
}

static memory addFc( const CDnn& dnn, const CString& fcName, memory& input, CDnnlNet& net )
{
	assert( dnn.HasLayer( fcName ) );
	const CFullyConnectedLayer& fc = *dynamic_cast<const CFullyConnectedLayer*>( dnn.GetLayer( fcName ).Ptr() );
	memory::dims srcDim = input.get_desc().dims();

	memory::desc srcMd = noFormat( input );
	memory::desc weightMd( { fc.GetNumberOfElements(), srcDim[1], 1, 1 }, memory::data_type::f32, memory::format_tag::any );
	memory::desc dstMd( { srcDim[0], fc.GetNumberOfElements() }, memory::data_type::f32, memory::format_tag::any );

	memory weightMemory( { { fc.GetNumberOfElements(), srcDim[1], 1, 1 }, memory::data_type::f32, memory::format_tag::nchw }, net.Engine() );
	copyToDnnlMemory( *fc.GetWeightsData(), weightMemory );

	if( fc.IsZeroFreeTerm() ) {
		inner_product_forward::primitive_desc ipPd( { prop_kind::forward_inference, srcMd, weightMd, dstMd }, net.Engine() );
		memory srcMemory = reorderIfNeeded( input, ipPd.src_desc(), fcName + "Src", net );
		weightMemory = reorderIfNeeded( weightMemory, ipPd.weights_desc(), fcName + "Weight", net );
		memory dstMemory( ipPd.dst_desc(), net.Engine() );
		net.AddPrimitive( inner_product_forward( ipPd ),
			{ { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_DST, dstMemory } }, fcName );
		return dstMemory;
	}

	memory::desc biasMd( { fc.GetNumberOfElements() }, memory::data_type::f32, memory::format_tag::any );
	memory biasMemory( { { fc.GetNumberOfElements() }, memory::data_type::f32, memory::format_tag::x }, net.Engine() );
	copyToDnnlMemory( *fc.GetFreeTermData(), biasMemory );

	inner_product_forward::primitive_desc ipPd( { prop_kind::forward_inference, srcMd, weightMd, biasMd, dstMd }, net.Engine() );
	memory srcMemory = reorderIfNeeded( input, ipPd.src_desc(), fcName + "Src", net );
	weightMemory = reorderIfNeeded( weightMemory, ipPd.weights_desc(), fcName + "Weight", net );
	biasMemory = reorderIfNeeded( biasMemory, ipPd.bias_desc(), fcName + "Bias", net );
	memory dstMemory( ipPd.dst_desc(), net.Engine() );
	net.AddPrimitive( inner_product_forward( ipPd ),
		{ { DNNL_ARG_SRC, srcMemory }, { DNNL_ARG_WEIGHTS, weightMemory }, { DNNL_ARG_BIAS, biasMemory }, { DNNL_ARG_DST, dstMemory } },
		fcName );
	return dstMemory;
}

static memory convertOutput( memory& currOutput, const std::string& name, CDnnlNet& net )
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
	return reorderIfNeeded( currOutput, preferredOutputDesc, name, net );
}

static void loadDnn( CDnn& dnn, const CString& netName )
{
	CArchiveFile file( netName + ".cnnarch", CArchive::load );
	CArchive archive( &file, CArchive::load );
	archive.Serialize( dnn );
	archive.Close();
	file.Close();
}

static memory buildDnnlNet( const CDnn& dnn, memory& input, CDnnlNet& net )
{
	memory output = addConv( dnn, "conv1", true, input, net );
	output = addBlock( dnn, "block0", output, net );
	output = addBlock( dnn, "block10", output, net );
	output = addBlock( dnn, "block11", output, net );
	output = addBlock( dnn, "block20", output, net );
	output = addBlock( dnn, "block21", output, net );
	output = addBlock( dnn, "block22", output, net );
	output = addBlock( dnn, "block30", output, net );
	output = addBlock( dnn, "block31", output, net );
	output = addBlock( dnn, "block32", output, net );
	output = addBlock( dnn, "block33", output, net );
	output = addBlock( dnn, "block40", output, net );
	output = addBlock( dnn, "block41", output, net );
	output = addBlock( dnn, "block42", output, net );
	output = addBlock( dnn, "block50", output, net );
	output = addBlock( dnn, "block51", output, net );
	output = addBlock( dnn, "block52", output, net );
	output = addBlock( dnn, "block6", output, net );
	output = addConv( dnn, "conv2", true, output, net );
	output = addMeanPooling( dnn, "pool", output, net );
	output = addFc( dnn, "fc", output, net );
	return convertOutput( output, "out", net );
}

static void adaptPoolSize( size_t originalSize, size_t actualSize, const CString& poolName, CDnn& dnn )
{
	assert( actualSize % originalSize == 0 );
	const size_t multiplier = actualSize / originalSize;
	if( multiplier != 1 ) {
		CPtr<CPoolingLayer> pool = CheckCast<CPoolingLayer>( dnn.GetLayer( poolName ) );
		pool->SetFilterHeight( static_cast<int>( pool->GetFilterHeight() * multiplier ) );
		pool->SetFilterWidth( static_cast<int>( pool->GetFilterWidth() * multiplier ) );
	}
}

static CDnnBlob* createInputBlob( IMathEngine& mathEngine, CRandom& random, const int imageSize, const int channelsCount )
{
	CDnnBlob* blob = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1, 1, imageSize, imageSize, channelsCount );
	float* buff = blob->GetBuffer<float>( 0, blob->GetDataSize() );
	for( int i = 0; i < blob->GetDataSize(); ++i ) {
		buff[i] = static_cast<float>( random.Uniform( -1, 2 ) );
	}
	blob->ReleaseBuffer( buff, true );
	return blob;
}

static IMathEngine* createMathEngine( engine::kind engineKind )
{
	if( engineKind == engine::kind::cpu ) {
		return CreateCpuMathEngine( 0, 0 );
	} else {
		std::unique_ptr<IGpuMathEngineManager> manager( CreateGpuMathEngineManager() );
		const int gpuMathEngineCount = manager->GetMathEngineCount();
		for( int i = 0; i < gpuMathEngineCount; ++i ) {
			CMathEngineInfo mathEngineInfo;
			manager->GetMathEngineInfo( i, mathEngineInfo );
			if( mathEngineInfo.Type == MET_Vulkan && std::string( mathEngineInfo.Name ).substr( 0, 5 ) == "Intel" ) {
				std::cout << "GPU:\t" << mathEngineInfo.Name << std::endl;
				return manager->CreateMathEngine( i, 0 );
				break;
			}
		}
	}
	return nullptr;
}

class CTestTimer {
public:
	CTestTimer( const std::string& header, IMathEngine& mathEngine );
	~CTestTimer();

private:
	std::string header;
	std::unique_ptr<IPerformanceCounters> counters;
};

CTestTimer::CTestTimer( const std::string& header, IMathEngine& mathEngine ) :
	header( header ), counters( mathEngine.CreatePerformanceCounters() )
{
	counters->Synchronise(); 
}

CTestTimer::~CTestTimer()
{
	try {
		counters->Synchronise(); 
		std::cout << header << std::endl;
		for( const auto& counter : *counters ) {
			std::cout << counter.Name << ": " << counter.Value << std::endl;
		}
	} catch( std::exception& ) {
	}
}

static std::vector<float> testDnnl( engine::kind engineKind, const CDnn& dnn, const CDnnBlob& inputBlob, size_t runCount, IMathEngine& mathEngine )
{
	CDnnlNet net( engineKind );
	memory::desc inputMd( { inputBlob.GetObjectCount(), inputBlob.GetChannelsCount(), inputBlob.GetHeight(), inputBlob.GetWidth() },
		memory::data_type::f32, memory::format_tag::nchw );
	memory input( inputMd, net.Engine() );
	copyToDnnlMemory( inputBlob, input );

	memory output = buildDnnlNet( dnn, input, net );

	const size_t outputBytes = output.get_desc().get_size();
	assert( outputBytes % sizeof( float ) == 0 );
	std::vector<float> result( outputBytes / sizeof( float ) );

	net.Execute();
	copyFromDnnlMemory( output, result );

	std::vector<float> buff( result.size() );
	{
		CTestTimer timer( "*** ONE DNN ***", dnn.GetMathEngine() );
		for( size_t run = 1; run <= runCount; ++run ) {
			engineKind == engine::kind::cpu ? net.Profile( mathEngine ) : net.Execute();
			copyFromDnnlMemory( output, buff );
		}
	}

	if( engineKind == engine::kind::cpu ) {
		const std::vector<std::string> primitives = net.PrimitiveNames();
		std::vector<TTimerType> times;
		std::transform( primitives.begin(), primitives.end(), std::back_inserter( times ),
			[&net] ( const std::string& name ) { return net.PrimitiveTime( name ); } );
		TTimerType total = std::accumulate( times.begin(), times.end(), 0 );
		::printf( "Perf stats:\n" );
		for( size_t i = 0; i < primitives.size(); ++i ) {
			if( times[i] == 0 ) {
				continue;
			}
			::printf( "%3d:%s\t%d\t(%f%%)\n", static_cast<int>( i ), primitives[i].data(),
				static_cast<int>( times[i] ), static_cast<float>( times[i] * 100. / total ) );
		}
		::printf( "\n" );
	}

	return result;
}

static std::vector<float> testNeoML( CDnn& dnn, CDnnBlob& inputBlob, const CString& inputName, const CString& outputName,
	size_t runCount )
{
	CPtr<CSourceLayer> in = CheckCast<CSourceLayer>( dnn.GetLayer( "in" ) );
	CPtr<CSinkLayer> out = CheckCast<CSinkLayer>( dnn.GetLayer( "out" ) );

	in->SetBlob( &inputBlob );
	dnn.RunOnce();
	const CPtr<CDnnBlob>& outputBlob = out->GetBlob();
	std::vector<float> result( outputBlob->GetDataSize() );
	float* buffer = outputBlob->GetBuffer<float>( 0, outputBlob->GetDataSize() );
	repackToChannelFirst( outputBlob->GetDesc(), buffer, result.data() );
	outputBlob->ReleaseBuffer( buffer, false );
	dnn.EnableProfile( true );

	std::vector<float> buff( outputBlob->GetDataSize() );
	{
		CTestTimer timer( "***  NeoML  ***", dnn.GetMathEngine() );
		for( size_t run = 1; run <= runCount; ++run ) {
			dnn.RunOnce();
			out->GetBlob()->CopyTo( buff.data() );
		}
	}

	if( dnn.GetMathEngine().GetType() == MET_Cpu ) {
		CArray<const char*> names;
		dnn.GetLayerList( names );
		std::vector<TTimerType> times;
		for( int i = 0; i < names.Size(); ++i ) {
			times.push_back( dnn.GetLayer( names[i] )->GetRunOnceTime() );
		}
		TTimerType total = std::accumulate( times.begin(), times.end(), 0 );
		::printf( "\nPerf stats:\n" );
		for( int i = 0; i < names.Size(); ++i ) {
			if( times[i] == 0 ) {
				continue;
			}
			::printf( "%3d:%s\t%d\t(%f%%)\n", static_cast<int>( i ), names[i],
				static_cast<int>( times[i] ), static_cast<float>( times[i] * 100. / total ) );
		}
		::printf( "\n" );
	}

	return result;
}

static float maxAbsError( const std::vector<float>& first, const std::vector<float>& second )
{
	assert( first.size() == second.size() );
	float result = 0;
	for( size_t i = 0; i < first.size(); ++i ) {
		result = max( result, fabsf( first[i] - second[i] ) );
	}
	return result;
}

int main( int argc, char** argv )
{
	try {
		const CString netName = "MobileNetV2Cifar10";
		const size_t runCount = 100;
		const int imageSize = 512;
		const int imageChannels = 3;
		engine::kind engineKind = engine::kind::cpu;

		std::unique_ptr<IMathEngine> mathEngine( createMathEngine( engineKind ) );
		if( mathEngine == nullptr ) {
			std::cerr << "Failed to create mathEngine" << std::endl;
			return 1;
		}

		CRandom random( 0x54 );
		CDnn dnn( random, *mathEngine );

		loadDnn( dnn, netName );

		CPtr<CDnnBlob> inputBlob = createInputBlob( *mathEngine, random, imageSize, imageChannels );
		adaptPoolSize( 32, imageSize, "pool", dnn );

		std::vector<float> actual = testDnnl( engineKind, dnn, *inputBlob, runCount, *mathEngine );
		std::vector<float> expected = testNeoML( dnn, *inputBlob, "in", "out", runCount );

		std::cout << "Max abs err: " << maxAbsError( expected, actual ) << std::endl;
	} catch( std::exception& ex ) {
		std::cerr << "Exception: " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
