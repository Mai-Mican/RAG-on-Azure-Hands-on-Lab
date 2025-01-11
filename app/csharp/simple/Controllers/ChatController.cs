using System;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using Simple.Services;

/// <summary>
/// チャット関連のリクエストを処理するコントローラー。
/// </summary>
[ApiController]
[Route("[controller]")]
public class ChatController : ControllerBase
{
    private readonly AoaiService _aoaiService;

    /// <summary>
    /// <see cref="ChatController"/> クラスの新しいインスタンスを初期化します。
    /// </summary>
    /// <param name="aoaiService">AOAIサービス。</param>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="aoaiService"/> がnullの場合にスローされます。
    /// </exception>
    public ChatController(AoaiService aoaiService)
    {
        _aoaiService = aoaiService ?? throw new ArgumentNullException(nameof(aoaiService));
    }

    /// <summary>
    /// チャットリクエストを処理します。
    /// </summary>
    /// <param name="query">チャットクエリ。</param>
    /// <returns>チャット応答を含む <see cref="IActionResult"/>。</returns>
    [HttpGet]
    public async Task<IActionResult> Chat(string query)
    {
        try
        {
            var response = await _aoaiService.GetChatResponseAsync(query);
            return Ok(response);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { error = ex.Message });
        }
    }
}