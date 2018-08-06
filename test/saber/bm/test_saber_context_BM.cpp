#include "test_saber_context_BM.h"

#ifdef USE_BM

using namespace anakin::saber;

TEST(TestSaberContextBM, test_BM_context) {
    Env<BM>::env_init();
    typedef TargetWrapper<BM> API;
    typename API::event_t event;
    API::create_event(event);
    LOG(INFO) << "test context constructor";
    Context<BM> ctx0;
    Context<BM> ctx1(0, 1, 1);
    LOG(INFO) << "test record event to context data stream and compute stream";
    API::record_event(event, ctx0.get_data_stream());
    API::record_event(event, ctx0.get_compute_stream());
    API::record_event(event, ctx1.get_data_stream());
    API::record_event(event, ctx1.get_compute_stream());
}

#endif

int main(int argc, const char** argv) {
    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

