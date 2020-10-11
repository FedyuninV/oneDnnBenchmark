#include <NeoML/NeoML.h>
#include <dnnl.hpp>
#include <memory>
#include <iostream>

class COneDnnOp {
public:
	virtual void Execute() = 0;
};

class COneDnnBlock : public COneDnnOp {
public:
	COneDnnBlock( CDnn& dnn, const CString& blockName, dnnl::engine& dengine, dnnl::stream& stream,
		dnnl::memory::desc& input );

private:
};

COneDnnBlock::COneDnnBlock( CDnn& dnn, const CString& blockName, dnnl::engine& dengine,
	dnnl::stream& stream, dnnl::memory::desc& input )
{
	CPtr<CConvLayer> conv1NeoML = CheckCast<CConvLayer>( dnn.GetLayer( blockName + "conv1" ) );

}

static void loadDnn( CDnn& dnn, const CString& netName, const CString& inputName )
{
	{
		CArchiveFile file( netName + ".cnnarch", CArchive::load );
		CArchive archive( &file, CArchive::load );
		archive.Serialize( dnn );
		archive.Close();
		file.Close();
	}

	{
		CPtr<CDnnBlob> inputBlob = new CDnnBlob( dnn.GetMathEngine() );
		CArchiveFile file( netName + "." + inputName + ".input", CArchive::load );
		CArchive archive( &file, CArchive::load );
		inputBlob->Serialize( archive );
		archive.Close();
		file.Close();
	}
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
			printf( "%s:\t%s\n", "Conv", layer->GetName() );
		} else if( dynamic_cast<const CChannelwiseConvLayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "ChannelwiseConv", layer->GetName() );
		} else if( dynamic_cast<const CReLULayer*>( layer ) != nullptr ) {
			printf( "%s:\t%s\n", "ReLU", layer->GetName() );
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
	std::unique_ptr<IMathEngine> mathEngine( CreateCpuMathEngine( 0, 0 ) );
	dnnl::engine eng( dnnl::engine::kind::cpu, 0 );

	CRandom random( 0x54 );
	CDnn dnn( random, *mathEngine );

	loadDnn( dnn, "MobileNetV2Cifar10", "in" );
	printNetInfo( dnn );

	return 0;
}
