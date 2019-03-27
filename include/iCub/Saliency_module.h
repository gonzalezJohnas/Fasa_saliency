//
// Created by jonas on 27/03/19.
//

#ifndef $KEYWORD_SALIENCY_MODULE_H
#define $KEYWORD_SALIENCY_MODULE_H

#include <yarp/os/RFModule.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/Semaphore.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Vocab.h>
#include <yarp/conf/numeric.h>
#include <yarp/sig/Image.h>
#include <yarp/sig/api.h>
#include "FASA.h"


// GENERAL VOCAB COMMAND
constexpr yarp::conf::vocab32_t COMMAND_VOCAB_OK = yarp::os::createVocab('o', 'k');

constexpr yarp::conf::vocab32_t COMMAND_VOCAB_SET = yarp::os::createVocab('s', 'e', 't');
constexpr yarp::conf::vocab32_t COMMAND_VOCAB_GET = yarp::os::createVocab('g', 'e', 't');
constexpr yarp::conf::vocab32_t COMMAND_VOCAB_RUN = yarp::os::createVocab('r', 'u', 'n');
constexpr yarp::conf::vocab32_t COMMAND_VOCAB_SUSPEND = yarp::os::createVocab('s', 'u', 's');
constexpr yarp::conf::vocab32_t COMMAND_VOCAB_RESUME = yarp::os::createVocab('r', 'e', 's');

constexpr yarp::conf::vocab32_t COMMAND_VOCAB_HELP = yarp::os::createVocab('h', 'e', 'l', 'p');
constexpr yarp::conf::vocab32_t COMMAND_VOCAB_FAILED = yarp::os::createVocab('f', 'a', 'i', 'l');



class Saliency_module : public yarp::os::RFModule {
public:


    /**
    *  configure all the objectTrackingModuleModule parameters and return true if successful
    * @param rf reference to the resource finder
    * @return flag for the success
    */
    bool configure(yarp::os::ResourceFinder &rf);

    /**
    *  interrupt, e.g., the ports
    */
    bool interruptModule();

    /**
    *  close and shut down
    */
    bool close();

    /**
    *  Respond through rpc port
    * @param command reference to bottle given to rpc port of module, along with parameters
    * @param reply reference to bottle returned by the rpc port in response to command
    * @return bool flag for the success of response else termination of module
    */

    bool respond(const yarp::os::Bottle &command, yarp::os::Bottle &reply);

    /**
    *  implemented to define the periodicity of the module
    */
    double getPeriod();

    /**
    *  Thread loop called periodically
    */
    bool updateModule();

private:
    std::string moduleName;
    std::string robotName;
    std::string handlerPortName;
    yarp::os::Port handlerPort;              // a port to handle messages
    yarp::os::Semaphore mutex;                  // semaphore for the respond function

    // Working port : Input image stream, output saliency
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > inputPort_imageStream;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelMono> > outputPort_saliency;

    yarp::sig::ImageOf<yarp::sig::PixelRgb> *inputImage_stream;
    yarp::sig::ImageOf<yarp::sig::PixelMono > *outputimage_saliency;

    // FASA saliency class
    Fasa fasa_saliency;

};


#endif //$KEYWORD_SALIENCY_MODULE_H
