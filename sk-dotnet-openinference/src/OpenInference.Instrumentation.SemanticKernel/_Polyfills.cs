#if NETSTANDARD2_0
namespace System.Runtime.CompilerServices
{
    using System.ComponentModel;

    /// <summary>
    /// Polyfill required for C# 9 records on netstandard2.0.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    internal static class IsExternalInit { }
}
#endif
