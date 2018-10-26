#include <arrow/tensor.h>
#include <arrow/stl.h>

#include "xtl/xdynamic_bitset.hpp"

#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xbuffer_adaptor.hpp"

namespace xt
{
    template <typename T>
    struct arrow_conversion {};

    #define XARROW_STL_CONVERSION(c_type, ArrowType_)             \
      template <>                                                 \
      struct arrow_conversion<c_type> {                           \
          using type = ArrowType_;                                \
          static std::shared_ptr<arrow::DataType> arrow_type() {  \
              return std::make_shared<ArrowType_>();              \
          }                                                       \
          constexpr static bool nullable = false;                 \
      };

    XARROW_STL_CONVERSION(bool, arrow::BooleanType)
    XARROW_STL_CONVERSION(int8_t, arrow::Int8Type)
    XARROW_STL_CONVERSION(int16_t, arrow::Int16Type)
    XARROW_STL_CONVERSION(int32_t, arrow::Int32Type)
    XARROW_STL_CONVERSION(int64_t, arrow::Int64Type)
    XARROW_STL_CONVERSION(uint8_t, arrow::UInt8Type)
    XARROW_STL_CONVERSION(uint16_t, arrow::UInt16Type)
    XARROW_STL_CONVERSION(uint32_t, arrow::UInt32Type)
    XARROW_STL_CONVERSION(uint64_t, arrow::UInt64Type)
    XARROW_STL_CONVERSION(float, arrow::FloatType)
    XARROW_STL_CONVERSION(double, arrow::DoubleType)
    XARROW_STL_CONVERSION(std::string, arrow::StringType)

    template <class T>
    class awoptional_assembly;

    template <class T>
    struct xcontainer_inner_types<awoptional_assembly<T>>
    {
        using value_expression = xt::xtensor_adaptor<xt::xbuffer_adaptor<T*, xt::no_ownership>, 1>;
        using value_storage_type = typename value_expression::storage_type&;
        using flag_expression = xt::xtensor_adaptor<xtl::xdynamic_bitset_view<uint8_t*>, 1>;
        using flag_storage_type = typename flag_expression::storage_type&;
        using storage_type = xoptional_assembly_storage<value_storage_type, flag_storage_type>;
        using temporary_type = awoptional_assembly<T>;
    };

    template <class T>
    struct xiterable_inner_types<awoptional_assembly<T>>
    {
        using assembly_type = awoptional_assembly<T>;
        using inner_shape_type = std::array<std::size_t, 1>;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    std::shared_ptr<arrow::Buffer> default_allocate_buffer(std::size_t sz)
    {
        std::shared_ptr<arrow::Buffer> buf;
        auto status = arrow::AllocateBuffer(arrow::default_memory_pool(), sz, &buf);
        if (!status.ok()) {
            throw std::runtime_error("Could not allocate Arrow memory.");
        }
        return buf;
    }

    std::shared_ptr<arrow::Buffer> default_allocate_flag(std::size_t sz)
    {
        std::shared_ptr<arrow::Buffer> buf;
        auto status = arrow::AllocateEmptyBitmap(arrow::default_memory_pool(), sz, &buf);
        if (!status.ok())
        {
            throw std::runtime_error("Could not allocate Arrow memory.");
        }
        return buf;
    }

    template <class T>
    class awoptional_assembly : public xoptional_assembly_base<awoptional_assembly<T>>,
                                public xcontainer_semantic<awoptional_assembly<T>>
    {
    public:

        using self_type = awoptional_assembly<T>;
        using base_type = xoptional_assembly_base<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_expression = typename base_type::value_expression;
        using flag_expression = typename base_type::flag_expression;
        using storage_type = typename base_type::storage_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using assembly_type = base_type;

        using valuebuffer_type = xt::xbuffer_adaptor<T*, xt::no_ownership>;
        using flagbuffer_type = xtl::xdynamic_bitset_view<uint8_t*>;

        using arrow_conversion = arrow_conversion<T>;
        using arrow_inner_type = typename arrow_conversion::type;
        using arrow_array = arrow::NumericArray<arrow_inner_type>;

        awoptional_assembly(std::initializer_list<value_type> init)
            : m_data(std::make_shared<arrow_array>(
                init.size(),
                default_allocate_buffer(init.size()),
                default_allocate_flag(init.size())
              )),
              m_value(valuebuffer_type((T*)(m_data->raw_values()), static_cast<std::size_t>(m_data->length())), {static_cast<std::size_t>(m_data->length())}),
              m_flag(flagbuffer_type((uint8_t*)m_data->null_bitmap_data(), static_cast<std::size_t>(m_data->length())), {static_cast<std::size_t>(m_data->length())}),
              m_storage(m_value.storage(), m_flag.storage())
        {
            std::copy(init.begin(), init.end(), this->begin());
        }

        awoptional_assembly(std::shared_ptr<arrow_array>& arr)
            : m_data(arr),
              m_value(valuebuffer_type((T*)(arr->raw_values()), static_cast<std::size_t>(arr->length())), {static_cast<std::size_t>(arr->length())}),
              m_flag(flagbuffer_type((uint8_t*)arr->null_bitmap_data(), static_cast<std::size_t>(arr->length())), {static_cast<std::size_t>(arr->length())}),
              m_storage(m_value.storage(), m_flag.storage())
        {
        }

        template <class E>
        awoptional_assembly(const xexpression<E>& expr)
            : m_data(std::make_shared<arrow_array>(
                expr.derived_cast().size(),
                default_allocate_buffer(expr.derived_cast().size()),
                default_allocate_flag(expr.derived_cast().size())
              )),
              m_value(valuebuffer_type((T*)(m_data->raw_values()), static_cast<std::size_t>(m_data->length())), {static_cast<std::size_t>(m_data->length())}),
              m_flag(flagbuffer_type((uint8_t*)m_data->null_bitmap_data(), static_cast<std::size_t>(m_data->length())), {static_cast<std::size_t>(m_data->length())}),
              m_storage(m_value.storage(), m_flag.storage())
        {
            semantic_base::assign(expr);
        }

        operator std::shared_ptr<arrow_array>()
        {
            return m_data;
        }

        const auto& storage_impl() const
        {
            return m_storage;
        }

        const auto& value_impl() const
        {
            return m_value;
        }

        const auto& has_value_impl() const
        {
            return m_flag;
        }

        auto& storage_impl()
        {
            return m_storage;
        }

        auto& value_impl()
        {
            return m_value;
        }

        auto& has_value_impl()
        {
            return m_flag;
        }

        std::shared_ptr<arrow_array> m_data;
        value_expression m_value;
        flag_expression m_flag;
        storage_type m_storage;
    };
}


