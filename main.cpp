#include <iostream>

#include <arrow/array.h>
#include <arrow/tensor.h>
#include <arrow/builder.h>

#include "xtensor/xio.hpp"

// #include "include/xarrow/awarray.hpp"
#include "include/xarrow/awcolumn.hpp"

using namespace arrow;

struct missing_tag
{
    template <class T, class B>
    constexpr operator xtl::xoptional<T, B>() const
    {
        return xtl::missing<T>();
    }
};

constexpr auto null_t = missing_tag{};

int main() {


    Status s;
    DoubleBuilder builder;

    s = builder.Append(1);
    s = builder.Append(2);
    s = builder.Append(3);
    s = builder.AppendNull();
    s = builder.Append(5);
    s = builder.Append(6);
    s = builder.Append(7);
    s = builder.Append(8);

    std::shared_ptr<Array> array;
    s = builder.Finish(&array);

    std::shared_ptr<NumericArray<DoubleType>> int64_array = std::static_pointer_cast<NumericArray<DoubleType>>(array);

    auto test = xt::awoptional_assembly<double>(int64_array);

    std::cout << test << std::endl;
    std::cout << test * test << std::endl;

    xt::awoptional_assembly<double> t2(test * test);

    std::cout << t2 << std::endl;

    std::shared_ptr<NumericArray<DoubleType>> t2AR = t2;

    std::cout << *t2AR << std::endl;

    xt::awoptional_assembly<double> t3({0.5, 12.32, null_t, 999.0});
    std::cout << t3 << std::endl;

    std::shared_ptr<NumericArray<DoubleType>> t3AR = t3;

    std::cout << *t3AR << std::endl;


    xt::awoptional_assembly<int8_t> t4 = xt::arange<int8_t>(0, 18);
    std::cout << *(std::shared_ptr<NumericArray<Int8Type>>)t4 << std::endl;

    // t4(3) = null_t; doesn't work yet .. need to investigate
    t4(3) = xtl::missing<int8_t>();
    std::cout << *(std::shared_ptr<NumericArray<Int8Type>>)t4 << std::endl;

}