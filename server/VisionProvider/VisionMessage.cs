namespace VisionProvider;

public class VisionMessage
{
    public string Method { get; set; }

    public byte[] Arguments { get; set; }

    public VisionMessage()
    {
        Method = string.Empty;
        Arguments = Array.Empty<byte>();
    }
}
