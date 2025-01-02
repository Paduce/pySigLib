#include "CppUnitTest.h"
#include "cusig.h"
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
namespace MyTest
{
    TEST_CLASS(sometests)
    {
    public:
        TEST_METHOD(exampletest)
        {
            Assert::IsTrue(true);
        }
    };
}