//
// Created by jonas on 27/03/19.
//

#include <iCub/Saliency_module.h>
#include <yarp/os/Log.h>
#include <yarp/cv/Cv.h>

bool Saliency_module::configure(yarp::os::ResourceFinder &rf) {
    using namespace yarp::os;


    if (rf.check("help")) {
        printf("HELP \n");
        printf("This a saliency module based on the FASA (Fast, Accurate, and Size-Aware Salient Object Detection) \n");
        printf("Take in input a video stream and output the saliency map) \n");
        printf("====== \n");
        printf("--name           : changes the rootname of the module ports \n");
        printf("--robot          : changes the name of the robot where the module interfaces to  \n");
        printf(" \n");
        printf("press CTRL-C to stop... \n");
        return true;
    }

    /* get the module name which will form the stem of all module port names */
    moduleName = rf.check("name",
                          Value("/FasaSaliency"),
                          "module name (string)").asString();
    /*
    * before continuing, set the module name before getting any other parameters,
    * specifically the port names which are dependent on the module name
    */
    setName(moduleName.c_str());

    /*
    * get the robot name which will form the stem of the robot ports names
    * and append the specific part and device required
    */
    robotName = rf.check("robot",
                         Value("icub"),
                         "Robot name (string)").asString();

    /*
    * attach a port of the same name as the module (prefixed with a /) to the module
    * so that messages received from the port are redirected to the respond method
    */
    handlerPortName = "";
    handlerPortName += getName();         // use getName() rather than a literal

    if (!handlerPort.open(handlerPortName.c_str())) {
        yInfo("%s : Unable to open port %s", getName().c_str(), handlerPortName.c_str() );
        return false;
    }

    if (!inputPort_imageStream.open(this->moduleName + "/inputImage:i")) {
        yInfo("%s : Unable to open port %s", getName().c_str(), (this->moduleName + "/inputImage:i").c_str() );
        return false;
    }

    if (!outputPort_saliency.open(this->moduleName + "/saliency:o")) {
        yInfo("%s : Unable to open port %s", getName().c_str(), (this->moduleName + "/inputImage:i").c_str() );
        return false;
    }


    attach(handlerPort);                  // attach to port

    return true;
}


bool Saliency_module::respond(const yarp::os::Bottle &command, yarp::os::Bottle &reply) {
    using namespace std;

    vector <string> replyScript;
    reply.clear();

    if (command.get(0).asString() == "quit") {
        reply.addString("quitting");
        this->stopModule();
        return false;
    }


    bool ok = false;
    bool rec = false; // is the command recognized?

    mutex.wait();

    switch (command.get(0).asVocab()) {

        case COMMAND_VOCAB_HELP:
            rec = true;
            {
                reply.addVocab(yarp::os::Vocab::encode("many"));
                reply.addString("Saliency module : Compute the saliency map using FASA algorithm");
                reply.addString("Other command :  ");




                ok = true;
            }
            break;

        case COMMAND_VOCAB_SET:
            rec = true;
            {
                switch (command.get(1).asVocab()) {

                    default:
                        yInfo("received an unknown request after SET");
                        break;
                }
            }
            break;

        case COMMAND_VOCAB_RUN:
            rec = true;
            {
                ok = true;
            }
            break;

        case COMMAND_VOCAB_GET:
            rec = true;
            {
                switch (command.get(1).asVocab()) {

                    default:
                        yInfo( "received an unknown request after a GET");
                        break;
                }
            }
            break;

        case COMMAND_VOCAB_SUSPEND:
            rec = true;
            {
                ok = true;
            }
            break;


        default:
            break;

    }
    mutex.post();

    if (!rec)
        ok = yarp::os::RFModule::respond(command, reply);

    if (!ok) {
        reply.clear();
        reply.addVocab(COMMAND_VOCAB_FAILED);
    } else
        reply.addVocab(COMMAND_VOCAB_OK);

    return ok;
}


bool Saliency_module::updateModule() {
    using namespace yarp::sig;
    while (!isStopping()) {
        inputImage_stream = inputPort_imageStream.read(false);

        if(inputImage_stream != nullptr){
            cv::Mat inputMat = yarp::cv::toCvMat(*inputImage_stream);
            cv::Mat saliencyMat;
            fasa_saliency.getSaliencyMap(inputMat, saliencyMat);

            outputimage_saliency = &outputPort_saliency.prepare();
            *outputimage_saliency = yarp::cv::fromCvMat<PixelMono >(saliencyMat);

            outputPort_saliency.write();


        }
    }
}



bool Saliency_module::close() {
    handlerPort.close();
    return true;
}

bool Saliency_module::interruptModule() {
    handlerPort.interrupt();
    return true;
}

double Saliency_module::getPeriod() {
    /* module periodicity (seconds), called implicitly by myModule */
    return 1.0;
}

