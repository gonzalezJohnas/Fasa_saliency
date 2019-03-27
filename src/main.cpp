#include <iCub/Saliency_module.h>
#include <yarp/os/Network.h>



int main(int argc, char *argv[]) {
    yarp::os::Network yarp;
    yarp::os::ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultConfigFile("FasaSaliency.ini");    //overridden by --from parameter
    rf.setDefaultContext("FasaSaliency");  //overridden by --context parameter
    rf.configure(argc, argv);
    Saliency_module module;
    return module.runModule(rf);


}
